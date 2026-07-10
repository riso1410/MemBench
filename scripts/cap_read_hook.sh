#!/bin/sh
# PreToolUse hook: cap whole-file Read calls at 400 lines to protect the
# 45k compact window. Rewrites Read tool_input.limit when missing/null or >400.
# Note: updatedInput fully REPLACES tool_input, so echo original input + limit.
input=$(cat)
tool=$(printf '%s' "$input" | jq -r '.tool_name // empty')
[ "$tool" = "Read" ] || exit 0
limit=$(printf '%s' "$input" | jq -r '.tool_input.limit // empty')
if [ -z "$limit" ] || [ "$limit" -gt 400 ] 2>/dev/null; then
  printf '%s' "$input" | jq -c '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"allow",updatedInput:(.tool_input+{limit:400})}}'
fi
exit 0
