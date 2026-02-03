using NoiaBot.Services;

namespace NoiaBot.Events;

public class HangupInputEvent(object sender) : EventBase(sender);
public class VolumeCtrlDownInputEvent(object sender) : EventBase(sender);
public class VolumeCtrlUpInputEvent(object sender) : EventBase(sender);