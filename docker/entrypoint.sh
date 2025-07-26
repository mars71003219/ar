#!/bin/bash
# entrypoint.sh

# X11 쿠키 동기화
if [ -f /root/.Xauthority ]; then
    xauth merge /root/.Xauthority
fi

# 나머지 명령어 실행
exec "$@"
