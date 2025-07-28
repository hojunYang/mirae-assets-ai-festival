#!/bin/bash

# 로그 파일 설정
LOG_DIR="/root/MA/log"
DATE=$(date +%Y%m%d)
LOG_FILE="$LOG_DIR/batch_direct_$DATE.log"

# 로그 디렉토리 생성
mkdir -p $LOG_DIR

# 로그 함수
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

# 간단한 서버 재시작 함수
restart_app_server() {
    # 기존 app.py 프로세스 찾아서 종료
    local app_pid=$(pgrep -f "python.*app.py")
    
    if [ -n "$app_pid" ]; then
        log "기존 app.py 프로세스 종료 중... (PID: $app_pid)"
        kill $app_pid
        sleep 2
    fi
    
    # 새로 시작
    log "새로운 app.py 서버 시작 중..."
    cd /root/MA
    nohup python app.py > /dev/null 2>&1 &
    log "app.py 서버 재시작 완료 (로그는 server_$(date +%Y%m%d).log 확인)"
}

log "=== 직접 Python 실행 배치 작업 시작 ==="

# 프로젝트 디렉토리로 이동
cd /root/MA

# 가상환경 활성화
source venv/bin/activate

# 방법 1: Python 코드를 직접 실행
log "Python 코드 직접 실행 중..."
python3 -c "
import sys
sys.path.append('/root/MA')

from datetime import date, timedelta
from batch import StockDataProcessor

# 현재 날짜 사용
today = date.today()

print(f'배치 작업 시작: {today.strftime(\"%Y-%m-%d\")}')
print(f'데이터 수집 기간: {today.strftime(\"%Y-%m-%d\")} ~ {today.strftime(\"%Y-%m-%d\")}')

try:
    processor = StockDataProcessor(today)
    processor.process_data(today, today)
    print('배치 작업 완료')
except Exception as e:
    print(f'배치 작업 중 오류 발생: {str(e)}')
    import traceback
    traceback.print_exc()
" >> $LOG_FILE 2>&1

# 실행 결과 확인
if [ $? -eq 0 ]; then
    log "배치 작업 성공적으로 완료"
    
    # 배치 성공시 app.py 서버 재시작
    log "app.py 서버 재시작 중..."
    restart_app_server
else
    log "배치 작업 실패 (종료 코드: $?)"
fi

# 방법 2: Python으로 시스템 정보 수집
log "시스템 정보 수집 중..."
SYSTEM_INFO=$(python3 -c "
import psutil
import os
from datetime import datetime

print(f'CPU 사용률: {psutil.cpu_percent()}%')
print(f'메모리 사용률: {psutil.virtual_memory().percent}%')
print(f'디스크 사용률: {psutil.disk_usage(\"/\").percent}%')
print(f'현재 시간: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
")

log "시스템 정보:"
log "$SYSTEM_INFO"

# 가상환경 비활성화
deactivate

log "=== 배치 작업 종료 ==="

echo "로그 파일: $LOG_FILE" 