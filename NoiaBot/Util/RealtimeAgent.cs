using Azure.AI.OpenAI;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.SemanticKernel;
using OpenAI.Realtime;
using Pv;
using System.ClientModel;
using System.Text;
namespace NoiaBot.Util;

public sealed class RealtimeAgentOptions
{
    public string Model { get; set; }
    public string Voice { get; set; }
    public string Instructions { get; set; }
    public string OpenAiApiKey { get; set; }
    public string OpenAiEndpoint { get; set; }
    public float? Temperature { get; set; }
    public int? ConversationInactivityTimeoutSeconds { get; set; }    
}

public enum RealtimeAgentRunResult
{
    Cancelled,
    InactivityTimeout
}

public enum StateUpdate
{
    Ready,
    SpeakingStarted,
    SpeakingStopped
}

public sealed class RealtimeAgent : IDisposable
{
    // Audio constants
    private const int SampleRate = 24000;
    private const int VadSampleRate = 16000;
    private const int FrameLength = 512;
    private const float VadThreshold = 0.5f;
    private const int MinSpeechFrames = 3;
    private const int MinSpeechFramesForBargeIn = 2;
    private const double SilenceMillisecondsToStop = 1600;
    private const int PreBufferFrames = 15;
    private const int SpeakerChunkSize = 4096;

    // Dependencies (immutable after construction)
    private readonly Kernel _kernel;
    private readonly ILogger _logger;
    private readonly RealtimeAgentOptions _options;

    // Session lifecycle
    private RealtimeClient _realtimeClient;
    private RealtimeSession _session;
    private CancellationTokenSource _sessionCts;
    private Task _receiveTask;
    private int _receiveGeneration;
    private bool _disposed;

    // Shared state between receive task and audio capture loop
    // _outputAudioLock protects all playback-state transitions and output audio buffering.
    // _speakerLock protects _currentSpeaker access.
    private readonly object _speakerLock = new();
    private readonly object _outputAudioLock = new();
    private readonly PlaybackSyncState _playbackState = new();
    private Speaker _currentSpeaker;
    private volatile Action<StateUpdate> _stateUpdateAction;

    public RealtimeAgent(ILogger<RealtimeAgent> logger, Kernel kernel, IOptions<RealtimeAgentOptions> options)
    {
        _logger = logger;
        _kernel = kernel;
        _options = options.Value;
    }

    // Public API

    /// <summary>
    /// Runs the conversation loop until cancellation is requested or VAD inactivity timeout occurs.
    /// Both cancellation and timeout preserve the session - call Dispose to close the session.
    /// </summary>
    public async Task<RealtimeAgentRunResult> RunAsync(
        Action<StateUpdate> stateUpdateAction = null, 
        Action<byte> meterAction = null, 
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        await EnsureSessionConnectedAsync(cancellationToken);

        using var recorder = PvRecorder.Create(frameLength: FrameLength, deviceIndex: -1);
        using var speaker = new Speaker(sampleRate: SampleRate, bitsPerSample: 16, meterAction: meterAction);
        using var vadDetector = new SileroVadDetector(VadSampleRate);

        lock (_speakerLock) { _currentSpeaker = speaker; }
        _logger.LogDebug($"Using microphone: {recorder.SelectedDevice}");

        EnsureReceiveTaskRunning();
        _stateUpdateAction = stateUpdateAction;
        ClearWaitingForResponse();
        stateUpdateAction?.Invoke(StateUpdate.Ready);

        var ctx = new AudioCaptureContext
        {
            Session = _session,
            Recorder = recorder,
            Speaker = speaker,
            VadDetector = vadDetector,
            SessionToken = _sessionCts!.Token,
            CancellationToken = cancellationToken
        };

        var result = await AudioCaptureLoopAsync(ctx);
        lock (_speakerLock) { _currentSpeaker = null; }
        return result;
    }

    public void Dispose()
    {
        if (_disposed) 
            return;
        _disposed = true;

        _sessionCts?.Cancel();

        if (_session is IDisposable disposable)
            disposable.Dispose();

        _sessionCts?.Dispose();
        _session = null;
        _sessionCts = null;
        _receiveTask = null;
    }

    // Session lifecycle

    private async Task EnsureSessionConnectedAsync(CancellationToken cancellationToken)
    {
        _realtimeClient ??= CreateRealtimeClient();

        if (_session is null)
        {
            _logger.LogDebug("Connecting to OpenAI Realtime API...");
            _sessionCts = new CancellationTokenSource();
            _session = await _realtimeClient.StartConversationSessionAsync(
                _options.Model, cancellationToken: cancellationToken);
            await ConfigureSessionAsync(_session, cancellationToken);
            _logger.LogDebug("Session configured.");
            return;
        }

        // If the receive task finished unexpectedly the WebSocket is dead - reconnect
        if (_receiveTask is not null && _receiveTask.IsCompleted)
        {
            _logger.LogDebug("[Session connection lost - reconnecting...]");
            await ResetSessionAsync();
            _sessionCts = new CancellationTokenSource();
            _session = await _realtimeClient.StartConversationSessionAsync(
                _options.Model, cancellationToken: cancellationToken);
            await ConfigureSessionAsync(_session, cancellationToken);
        }
    }

    private void EnsureReceiveTaskRunning()
    {
        if (_receiveTask is not null && !_receiveTask.IsCompleted)
            return;

        int generation = System.Threading.Interlocked.Increment(ref _receiveGeneration);
        _receiveTask = RunReceiveTaskAsync(_session, _sessionCts!.Token, generation);
    }

    private async Task ResetSessionAsync()
    {
        if (_sessionCts is not null)
            await _sessionCts.CancelAsync();

        if (_receiveTask is not null)
        {
            try
            {
                var completed = await Task.WhenAny(_receiveTask, Task.Delay(TimeSpan.FromSeconds(5)));
                if (!ReferenceEquals(completed, _receiveTask))
                    _logger.LogWarning("[ResetSession timed out waiting for receive task - continuing cleanup]");
            }
            catch
            {
                // Ignore errors during cleanup
            }
        }

        if (_session is IAsyncDisposable asyncDisposable)
        {
            try { await asyncDisposable.DisposeAsync(); }
            catch { /* ignore during cleanup */ }
        }
        else if (_session is IDisposable disposable)
        {
            disposable.Dispose();
        }

        _sessionCts?.Dispose();
        _session = null;
        _sessionCts = null;
        _receiveTask = null;
    }

    private async Task ConfigureSessionAsync(RealtimeSession session, CancellationToken cancellationToken)
    {
        var sessionOptions = new ConversationSessionOptions()
        {
            Voice = _options.Voice,
            InputAudioFormat = RealtimeAudioFormat.Pcm16,
            OutputAudioFormat = RealtimeAudioFormat.Pcm16,
            Instructions = _options.Instructions,
            TurnDetectionOptions = TurnDetectionOptions.CreateDisabledTurnDetectionOptions(),
            Temperature = _options.Temperature
        };

        if (_kernel != null)
        {
            foreach (var tool in KernelToolInvoker.ConvertKernelFunctions(_kernel))
            {
                _logger.LogDebug($"[Adding tool: {tool.Name}: {tool.Description}]");
                sessionOptions.Tools.Add(tool);
            }

            if (sessionOptions.Tools.Count > 0)
                sessionOptions.ToolChoice = ConversationToolChoice.CreateAutoToolChoice();
        }

        await session.ConfigureConversationSessionAsync(sessionOptions, cancellationToken);
    }

    private RealtimeClient CreateRealtimeClient()
    {
        var apiKey = _options.OpenAiApiKey;
        var endpoint = _options.OpenAiEndpoint;

        if (!string.IsNullOrWhiteSpace(apiKey) && string.IsNullOrWhiteSpace(endpoint))
            return new RealtimeClient(new ApiKeyCredential(apiKey));

        if (!string.IsNullOrWhiteSpace(apiKey) && !string.IsNullOrWhiteSpace(endpoint))
        {
            var client = new AzureOpenAIClient(
                endpoint: new Uri(endpoint),
                credential: new ApiKeyCredential(apiKey));
            return client.GetRealtimeClient();
        }

        throw new InvalidOperationException(
            "OpenAI/Azure OpenAI configuration was not found. " +
            "Please set OpenAiApiKey and optionally OpenAiEndpoint.");
    }

    // Receive loop - processes server updates for the session lifetime

    private async Task RunReceiveTaskAsync(RealtimeSession session, CancellationToken sessionToken, int receiveGeneration)
    {
        var ctx = new ReceiveContext(session, sessionToken);

        try
        {
            await foreach (RealtimeUpdate update in session.ReceiveUpdatesAsync(sessionToken))
            {
                if (sessionToken.IsCancellationRequested) break;
                if (receiveGeneration != System.Threading.Volatile.Read(ref _receiveGeneration)) break;
                await DispatchReceiveUpdateAsync(update, ctx);
            }
        }
        catch (OperationCanceledException)
        {
            // Expected when session is cancelled (dispose)
        }
        catch (Exception ex)
        {
            _logger.LogDebug($"[Receive task error: {ex.Message}]");
        }
    }

    private async Task DispatchReceiveUpdateAsync(RealtimeUpdate update, ReceiveContext ctx)
    {
        switch (update)
        {
            case ConversationSessionStartedUpdate started:
                _logger.LogDebug($"[Session started: {started.SessionId}]");
                break;
            case OutputStreamingStartedUpdate streamStart:
                HandleStreamingStarted(streamStart, ctx);
                break;
            case OutputDeltaUpdate delta:
                HandleOutputDelta(delta, ctx);
                break;
            case OutputStreamingFinishedUpdate streamEnd:
                await HandleStreamingFinishedAsync(streamEnd, ctx);
                break;
            case InputAudioTranscriptionFinishedUpdate transcription:
                _logger.LogDebug($"[You said: {transcription.Transcript}]");
                break;
            case ResponseFinishedUpdate responseEnd:
                await HandleResponseFinishedAsync(responseEnd, ctx);
                break;
            case RealtimeErrorUpdate error:
                _logger.LogError($"[Error: {error.Message}]");
                break;
        }
    }

    private void HandleStreamingStarted(OutputStreamingStartedUpdate update, ReceiveContext ctx)
    {
        _stateUpdateAction?.Invoke(StateUpdate.SpeakingStarted);

        lock (_outputAudioLock)
        {
            _playbackState.WaitingForResponse = false;
            _playbackState.ResponseRequestedAtUtc = default;
            _playbackState.ModelIsSpeaking = true;
            _playbackState.BargeInTriggered = false;
            _playbackState.CurrentStreamingItemId = update.ItemId;
            ctx.OutputAudioBuffer.Clear();
        }

        _logger.LogDebug($"[OutputStreamingStarted: FunctionName={update.FunctionName ?? "null"}, ItemId={update.ItemId ?? "null"}]");

        if (!string.IsNullOrEmpty(update.FunctionName))
            _logger.LogDebug($"[Calling: {update.FunctionName}] ");
    }

    private void HandleOutputDelta(OutputDeltaUpdate delta, ReceiveContext ctx)
    {
        if (!string.IsNullOrEmpty(delta.AudioTranscript))
            _logger.LogDebug(delta.AudioTranscript);

        if (!string.IsNullOrEmpty(delta.Text))
            _logger.LogDebug($"[TextDelta: {delta.Text}]");

        BufferOutputAudio(delta, ctx);
        AccumulateFunctionArguments(delta, ctx);
    }

    private void BufferOutputAudio(OutputDeltaUpdate delta, ReceiveContext ctx)
    {
        if (delta.AudioBytes is null)
            return;

        lock (_outputAudioLock)
        {
            if (_playbackState.BargeInTriggered)
                return;

            var audioBytes = delta.AudioBytes.ToArray();
            ctx.OutputAudioBuffer.AddRange(audioBytes);

            while (ctx.OutputAudioBuffer.Count >= SpeakerChunkSize)
            {
                byte[] chunk = new byte[SpeakerChunkSize];
                ctx.OutputAudioBuffer.CopyTo(0, chunk, 0, SpeakerChunkSize);
                ctx.OutputAudioBuffer.RemoveRange(0, SpeakerChunkSize);
                WriteSpeakerSafe(chunk);
            }
        }
    }

    private void AccumulateFunctionArguments(OutputDeltaUpdate delta, ReceiveContext ctx)
    {
        if (string.IsNullOrWhiteSpace(delta.FunctionArguments))
            return;

        if (delta.ItemId is null)
            return;

        if (!ctx.FunctionArgumentBuilders.TryGetValue(delta.ItemId, out var builder))
            ctx.FunctionArgumentBuilders[delta.ItemId] = builder = new StringBuilder();

        builder.Append(delta.FunctionArguments);
        _logger.LogDebug($"[FunctionArgsDelta: ItemId={delta.ItemId}, FunctionCallId={delta.FunctionCallId ?? "null"}, Args={delta.FunctionArguments}]");
    }

    private async Task HandleStreamingFinishedAsync(OutputStreamingFinishedUpdate update, ReceiveContext ctx)
    {
        _logger.LogDebug($"[OutputStreamingFinished: FunctionCallId={update.FunctionCallId ?? "null"}, FunctionName={update.FunctionName ?? "null"}, ItemId={update.ItemId ?? "null"}]");

        if (update.FunctionCallId is null)
            return;

        if (_kernel == null)
        {
            throw new InvalidOperationException(
                $"Function '{update.FunctionName}' was called but kernel is null. " +
                "A kernel must be provided when tools are configured in the session.");
        }

        _logger.LogDebug($"[Executing function: {update.FunctionName}]");

        RealtimeItem functionOutputItem;
        try
        {
            functionOutputItem = await KernelToolInvoker.InvokeFunctionAsync(
                update.FunctionName, update.FunctionCallId, update.ItemId,
                ctx.FunctionArgumentBuilders, _kernel, ctx.SessionToken);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"[Function '{update.FunctionName}' failed: {ex.Message}]");
            functionOutputItem = RealtimeItem.CreateFunctionCallOutput(
                callId: update.FunctionCallId,
                output: $"Error: {ex.Message}");
        }
        finally
        {
            ctx.FunctionArgumentBuilders.Remove(update.ItemId);
        }

        await ctx.Session.AddItemAsync(functionOutputItem, ctx.SessionToken);
    }

    private async Task HandleResponseFinishedAsync(ResponseFinishedUpdate update, ReceiveContext ctx)
    {
        ClearWaitingForResponse();
        FlushRemainingOutputAudio(ctx);

        await FlushSpeakerSafeAsync(ctx.SessionToken);
        SetModelSpeaking(false);
        _stateUpdateAction?.Invoke(StateUpdate.SpeakingStopped);

        _logger.LogDebug($"[ResponseFinished: {update.CreatedItems.Count} items created]");
        foreach (var item in update.CreatedItems)
            _logger.LogDebug($"  - Item: FunctionName={item.FunctionName ?? "null"}, FunctionCallId={item.FunctionCallId ?? "null"}, MessageRole={item.MessageRole}");

        if (update.CreatedItems.Any(item => item.FunctionName?.Length > 0))
        {
            _logger.LogDebug("[Function calls detected - triggering response...]");
            SetWaitingForResponse();
            await ctx.Session.StartResponseAsync(ctx.SessionToken);
        }
        else
        {
            _logger.LogDebug("[Ready for your next question...]");
        }
    }

    private void FlushRemainingOutputAudio(ReceiveContext ctx)
    {
        lock (_outputAudioLock)
        {
            if (ctx.OutputAudioBuffer.Count == 0 || _playbackState.BargeInTriggered)
                return;

            byte[] remainingChunk = new byte[ctx.OutputAudioBuffer.Count];
            ctx.OutputAudioBuffer.CopyTo(remainingChunk, 0);
            WriteSpeakerSafe(remainingChunk);
            ctx.OutputAudioBuffer.Clear();
        }
    }

    // Audio capture loop - local VAD, recording, barge-in

    private async Task<RealtimeAgentRunResult> AudioCaptureLoopAsync(AudioCaptureContext ctx)
    {
        ctx.Recorder.Start();
        ctx.Speaker.Start();

        try
        {
            while (!ctx.CancellationToken.IsCancellationRequested)
            {
                var frame = ctx.Recorder.Read();
                double frameDurationMs = (frame.Length * 1000.0) / ctx.Recorder.SampleRate;
                bool isSpeech = DetectSpeech(frame, ctx.Recorder.SampleRate, ctx.VadDetector);
                UpdateActivityTracking(ctx.State, isSpeech);

                await TryHandleBargeInAsync(isSpeech, ctx);
                MaintainPreBuffer(frame, ctx.Recorder.SampleRate, ctx.State);
                await ProcessRecordingStateAsync(frame, frameDurationMs, isSpeech, ctx);
                EnforceResponseWaitTimeout();

                if (IsInactivityTimeoutReached(ctx.State))
                {
                    _logger.LogDebug("[Inactivity timeout - pausing audio capture...]");
                    return RealtimeAgentRunResult.InactivityTimeout;
                }

                await Task.Delay(1, ctx.CancellationToken);
            }
        }
        catch (OperationCanceledException)
        {
            // Expected when cancellation is requested - session stays alive
        }
        finally
        {
            ctx.Recorder.Stop();
            ctx.Speaker.Stop();
        }

        return RealtimeAgentRunResult.Cancelled;
    }

    private static bool DetectSpeech(short[] frame, int recorderSampleRate, SileroVadDetector vadDetector)
    {
        var vadFrame = AudioResampler.DownsampleForVad(frame, recorderSampleRate, VadSampleRate, FrameLength);
        var floatFrame = new float[vadFrame.Length];

        for (int i = 0; i < vadFrame.Length; i++)
            floatFrame[i] = vadFrame[i] / 32768.0f;

        return vadDetector.Process(floatFrame) >= VadThreshold;
    }

    private void UpdateActivityTracking(AudioCaptureState state, bool isSpeech)
    {
        bool modelIsSpeaking = IsModelSpeaking();

        if (isSpeech)
            state.LastActivityUtc = DateTime.UtcNow;

        if (state.WasModelSpeaking && !modelIsSpeaking)
            state.LastActivityUtc = DateTime.UtcNow;

        state.WasModelSpeaking = modelIsSpeaking;
    }

    private async Task TryHandleBargeInAsync(bool isSpeech, AudioCaptureContext ctx)
    {
        var state = ctx.State;

        if (!IsModelSpeaking())
        {
            state.BargeInSpeechFrameCount = 0;
            return;
        }

        if (!isSpeech)
        {
            state.BargeInSpeechFrameCount = 0;
            return;
        }

        state.BargeInSpeechFrameCount++;
        if (state.BargeInSpeechFrameCount < MinSpeechFramesForBargeIn)
            return;

        if (!TryStartBargeIn(out string truncateItemId))
            return;

        int audioEndMs = ctx.Speaker.GetEstimatedPlayedMilliseconds();
        _logger.LogWarning($"[Barge-in detected - interrupting model at {audioEndMs}ms, ItemId={truncateItemId}]");
        ClearSpeakerSafe();

        try
        {
            await ctx.Session.CancelResponseAsync(ctx.SessionToken);
        }
        catch (Exception ex)
        {
            _logger.LogDebug($"[Cancel response failed (expected if generation already finished): {ex.Message}]");
        }

        if (truncateItemId is not null)
        {
            try
            {
                await ctx.Session.TruncateItemAsync(truncateItemId, 0, TimeSpan.FromMilliseconds(audioEndMs), ctx.SessionToken);
            }
            catch (Exception ex)
            {
                _logger.LogDebug($"[Truncate failed: {ex.Message}]");
            }
        }

        SetModelSpeaking(false);
        _stateUpdateAction?.Invoke(StateUpdate.SpeakingStopped);
        state.IsRecording = true;
        state.AudioBuffer.Clear();
        state.SpeechFrameCount = 0;
        state.BargeInSpeechFrameCount = 0;
        state.SilenceDurationMs = 0;
        ctx.VadDetector.Reset();
    }

    private static void MaintainPreBuffer(short[] frame, int recorderSampleRate, AudioCaptureState state)
    {
        if (state.IsRecording)
            return;

        var upsampledFrame = AudioResampler.UpsampleTo24kHz(frame, recorderSampleRate);
        state.PreBuffer.Enqueue(upsampledFrame);

        while (state.PreBuffer.Count > PreBufferFrames)
            state.PreBuffer.Dequeue();
    }

    private async Task ProcessRecordingStateAsync(
        short[] frame, double frameDurationMs, bool isSpeech, AudioCaptureContext ctx)
    {
        var state = ctx.State;

        if (!state.IsRecording && !IsModelSpeaking())
        {
            TryStartRecording(isSpeech, state);
            return;
        }

        if (!state.IsRecording)
            return;

        await ContinueOrStopRecordingAsync(frame, frameDurationMs, isSpeech, ctx);
    }

    private void TryStartRecording(bool isSpeech, AudioCaptureState state)
    {
        if (!isSpeech)
        {
            state.SpeechFrameCount = 0;
            return;
        }

        state.SpeechFrameCount++;
        if (state.SpeechFrameCount < MinSpeechFrames)
            return;

        _logger.LogDebug("[Voice detected - recording...]");
        state.IsRecording = true;
        state.AudioBuffer.Clear();

        while (state.PreBuffer.Count > 0)
            state.AudioBuffer.AddRange(state.PreBuffer.Dequeue());

        state.SilenceDurationMs = 0;
    }

    private async Task ContinueOrStopRecordingAsync(
        short[] frame, double frameDurationMs, bool isSpeech, AudioCaptureContext ctx)
    {
        var state = ctx.State;
        var upsampledFrame = AudioResampler.UpsampleTo24kHz(frame, ctx.Recorder.SampleRate);
        state.AudioBuffer.AddRange(upsampledFrame);

        if (isSpeech)
        {
            state.SilenceDurationMs = 0;
            return;
        }

        state.SilenceDurationMs += frameDurationMs;
        if (state.SilenceDurationMs < SilenceMillisecondsToStop)
            return;

        _logger.LogDebug("[Silence detected - sending to model...]");

        var audioBytes = AudioResampler.ShortsToBytes(state.AudioBuffer.ToArray());
        await ctx.Session.SendInputAudioAsync(new MemoryStream(audioBytes), ctx.CancellationToken);
        await ctx.Session.CommitPendingAudioAsync(ctx.CancellationToken);
        await ctx.Session.StartResponseAsync(ctx.CancellationToken);
        SetWaitingForResponse();

        state.IsRecording = false;
        state.AudioBuffer.Clear();
        state.SilenceDurationMs = 0;
        state.SpeechFrameCount = 0;
        ctx.VadDetector.Reset();
    }

    private void EnforceResponseWaitTimeout()
    {
        if (!TryExpireWaitingForResponse(TimeSpan.FromSeconds(30)))
            return;

        _logger.LogWarning("[Response wait timeout - model did not respond within 30s]");
    }

    private bool IsInactivityTimeoutReached(AudioCaptureState state)
    {
        GetModelAndWaitingState(out bool modelIsSpeaking, out bool waitingForResponse);
        if (state.IsRecording || modelIsSpeaking || waitingForResponse)
            return false;

        var timeout = TimeSpan.FromSeconds(_options.ConversationInactivityTimeoutSeconds ?? 10);
        return DateTime.UtcNow - state.LastActivityUtc >= timeout;
    }

    private void GetModelAndWaitingState(out bool modelIsSpeaking, out bool waitingForResponse)
    {
        lock (_outputAudioLock)
        {
            modelIsSpeaking = _playbackState.ModelIsSpeaking;
            waitingForResponse = _playbackState.WaitingForResponse;
        }
    }

    private bool IsModelSpeaking()
    {
        lock (_outputAudioLock)
        {
            return _playbackState.ModelIsSpeaking;
        }
    }

    private void SetModelSpeaking(bool modelIsSpeaking)
    {
        lock (_outputAudioLock)
        {
            _playbackState.ModelIsSpeaking = modelIsSpeaking;
        }
    }

    private void SetWaitingForResponse()
    {
        lock (_outputAudioLock)
        {
            _playbackState.WaitingForResponse = true;
            _playbackState.ResponseRequestedAtUtc = DateTime.UtcNow;
        }
    }

    private void ClearWaitingForResponse()
    {
        lock (_outputAudioLock)
        {
            _playbackState.WaitingForResponse = false;
            _playbackState.ResponseRequestedAtUtc = default;
        }
    }

    private bool TryExpireWaitingForResponse(TimeSpan maxWait)
    {
        lock (_outputAudioLock)
        {
            if (!_playbackState.WaitingForResponse)
            {
                return false;
            }

            if (DateTime.UtcNow - _playbackState.ResponseRequestedAtUtc <= maxWait)
            {
                return false;
            }

            _playbackState.WaitingForResponse = false;
            _playbackState.ResponseRequestedAtUtc = default;
            return true;
        }
    }

    private bool TryStartBargeIn(out string truncateItemId)
    {
        lock (_outputAudioLock)
        {
            if (!_playbackState.ModelIsSpeaking)
            {
                truncateItemId = null;
                return false;
            }

            _playbackState.BargeInTriggered = true;
            truncateItemId = _playbackState.CurrentStreamingItemId;
            return true;
        }
    }

    // Thread-safe speaker access

    private void WriteSpeakerSafe(byte[] data)
    {
        lock (_speakerLock) { _currentSpeaker?.Write(data); }
    }

    private void ClearSpeakerSafe()
    {
        lock (_speakerLock) { _currentSpeaker?.Clear(); }
    }

    private async Task FlushSpeakerSafeAsync(CancellationToken cancellationToken)
    {
        Speaker speaker;
        lock (_speakerLock) { speaker = _currentSpeaker; }

        if (speaker != null)
            await speaker.FlushAsync(cancellationToken);
    }

    // Context / state types for the two concurrent loops

    private sealed class PlaybackSyncState
    {
        public bool ModelIsSpeaking { get; set; }
        public bool WaitingForResponse { get; set; }
        public DateTime ResponseRequestedAtUtc { get; set; }
        public bool BargeInTriggered { get; set; }
        public string CurrentStreamingItemId { get; set; }
    }

    /// <summary>
    /// Bundles dependencies and per-loop state for the receive loop,
    /// so handlers don't need many individual parameters.
    /// </summary>
    private sealed class ReceiveContext
    {
        public ReceiveContext(RealtimeSession session, CancellationToken sessionToken)
        {
            Session = session;
            SessionToken = sessionToken;
        }

        public RealtimeSession Session { get; }
        public CancellationToken SessionToken { get; }
        public List<byte> OutputAudioBuffer { get; } = new();
        public Dictionary<string, StringBuilder> FunctionArgumentBuilders { get; } = new();
    }

    /// <summary>
    /// Bundles dependencies and per-loop state for the audio capture loop,
    /// so handlers don't need many individual parameters.
    /// </summary>
    private sealed class AudioCaptureContext
    {
        public required RealtimeSession Session { get; init; }
        public required PvRecorder Recorder { get; init; }
        public required Speaker Speaker { get; init; }
        public required SileroVadDetector VadDetector { get; init; }
        public required CancellationToken SessionToken { get; init; }
        public required CancellationToken CancellationToken { get; init; }
        public AudioCaptureState State { get; } = new();
    }

    /// <summary>
    /// Mutable state for one audio capture loop run (not shared across threads).
    /// </summary>
    private sealed class AudioCaptureState
    {
        public List<short> AudioBuffer { get; } = new();
        public Queue<short[]> PreBuffer { get; } = new();
        public bool IsRecording { get; set; }
        public int SpeechFrameCount { get; set; }
        public int BargeInSpeechFrameCount { get; set; }
        public double SilenceDurationMs { get; set; }
        public bool WasModelSpeaking { get; set; }
        public DateTime LastActivityUtc { get; set; } = DateTime.UtcNow;
    }
}
