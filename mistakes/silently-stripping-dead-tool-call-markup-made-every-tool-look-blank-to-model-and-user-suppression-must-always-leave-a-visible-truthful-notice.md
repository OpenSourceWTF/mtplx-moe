# silently-stripping-dead-tool-call-markup-made-every-tool-look-blank-to-model-and-user-suppression-must-always-leave-a-visible-truthful-notice

**Symptom:** Fresh-install user (issue #349) asks the built-in chat about
`~/Dev`; the model "invokes" `ls`/`find`/`search_files`/`read_file` and gets
NOTHING back — no output, no error. The model itself reports "tool calls
going out into the void"; the user sees blank replies and files a
first-run-experience bug. TCC/sandbox and the Settings working folder were
innocent.

**Cause:** The #160 fix hid raw `<tool_call>` XML from no-tools chats by
DELETING it (`_strip_orphan_tool_markup` + the stream splitter's orphan
filter), and the malformed-parse fallback did the same for undeclared tool
names. Deletion was silent: nothing executed, nothing said so, and the
replayed history contained neither the call nor a result — so the model
retried tools forever and the user saw blanks. One bug (raw XML shown) was
fixed by masking a worse one (calls swallowed without a trace).

**Fix / rule:** Suppression is never silent. Whenever tool markup is
stripped without execution, append a visible truthful notice (names the
tool, states nothing ran, says how to get real file/terminal access) so
both the user and the model's next-turn history carry the truth — the model
then self-corrects instead of spiraling. Stamp `unexecuted_tool_call_notice`
in stats. Client side: every tool call that reaches the transcript must have
a non-empty result message — a persisted `tool_calls` turn with no results
is a protocol violation the model reads as "the tool returned nothing".
When a fix hides something from the user, always ask what the MODEL will
see in the replayed transcript afterwards.
