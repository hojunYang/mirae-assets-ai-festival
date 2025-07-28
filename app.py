import os
import logging
from datetime import datetime
from typing import Optional, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from chatbot import MiraeAIFestivalchatbot
from tools import Tools

# 프로젝트 설정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJECT_ROOT, "log")
DATE_FORMAT = "%Y%m%d"

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"{LOG_DIR}/server_{datetime.now().strftime(DATE_FORMAT)}.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)

# 시스템 프롬프트 로드
with open("clova.prompt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# CLOVA Studio 설정
CLOVASTUDIO_ENDPOINT = (
    "https://clovastudio.stream.ntruss.com/testapp/v3/chat-completions/HCX-005"
)

# 최대 대화 메시지 수
MAX_CONVERSATION_LENGTH = 12

# 전역 변수 - 챗봇 인스턴스 관리
chatbot_instances: Dict[str, MiraeAIFestivalchatbot] = {}
tools_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작시 Tools 인스턴스 초기화"""
    global tools_instance
    
    logger.info("미래에셋 AI Festival 챗봇 서버 시작")
    
    try:
        # Tools 인스턴스 생성 (전역으로 하나만)
        tools_instance = Tools(logger)
        logger.info("Tools 인스턴스 생성 완료")
        
    except Exception as e:
        logger.error(f"Tools 초기화 실패: {str(e)}")
        raise Exception(f"Tools 초기화에 실패했습니다: {str(e)}")
    
    yield  # 애플리케이션 실행
    
    # 종료시 정리 작업
    chatbot_instances.clear()
    logger.info("서버 종료")

# FastAPI 앱 생성
app = FastAPI(
    title="미래에셋 AI Festival 챗봇 API",
    description="한국 주식시장 전문 AI 챗봇 서버",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_chatbot_key(api_key: str, request_id: str) -> str:
    """API 키와 Request ID를 조합한 챗봇 식별 키 생성"""
    return f"{api_key[:10]}...#{request_id}"

def get_or_create_chatbot(api_key: str, request_id: str) -> MiraeAIFestivalchatbot:
    """API 키와 Request ID에 따른 챗봇 인스턴스 생성 또는 조회"""
    global chatbot_instances, tools_instance
    
    chatbot_key = get_chatbot_key(api_key, request_id)
    
    if chatbot_key not in chatbot_instances:
        logger.info(f"새로운 챗봇 인스턴스 생성: {chatbot_key}")
        chatbot_instances[chatbot_key] = MiraeAIFestivalchatbot(
            api_key=api_key,
            request_id=request_id,
            endpoint=CLOVASTUDIO_ENDPOINT,
            logger=logger,
            tools=tools_instance
        )
    
    return chatbot_instances[chatbot_key]

def manage_conversation_length(chatbot: MiraeAIFestivalchatbot):
    """대화 길이 관리 - 30건 넘으면 초기화"""
    if len(chatbot.conversation_history) > MAX_CONVERSATION_LENGTH:
        logger.info(f"대화 메시지가 {len(chatbot.conversation_history)}건으로 {MAX_CONVERSATION_LENGTH}건을 초과하여 초기화")
        chatbot.clear_conversation()

@app.get("/")
async def root():
    """서버 상태 확인"""
    return {
        "message": "미래에셋 AI Festival 챗봇 서버가 실행 중입니다",
        "version": "1.0.0",
        "status": "healthy",
        "active_chatbots": len(chatbot_instances)
    }

@app.get("/health")
async def health_check():
    """헬스 체크"""
    global tools_instance
    
    if tools_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tools가 초기화되지 않았습니다"
        )
    
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "active_chatbots": len(chatbot_instances)
    }

@app.get("/agent")
async def agent(
    question: str = Query(..., description="질문"),
    authorization: Optional[str] = Header(None),
    x_ncp_clovastudio_request_id: Optional[str] = Header(None)
):
    """주식 질문 처리 API"""
    
    # API 키 추출
    api_key = None
    if authorization:
        if authorization.startswith("Bearer "):
            api_key = authorization[7:]  # "Bearer " 제거
        else:
            api_key = authorization
    
    if not api_key:
        api_key = os.getenv("CLOVASTUDIO_API_KEY")
    
    # Request ID 추출
    request_id = x_ncp_clovastudio_request_id or os.getenv("CLOVASTUDIO_REQUEST_ID") or "default-request-id"
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API 키가 필요합니다. Authorization 헤더 또는 환경변수로 제공해주세요."
        )
    
    try:
        # 요청 로깅
        chatbot_key = get_chatbot_key(api_key, request_id)
        logger.info(f"질문 요청 수신: {question[:100]}..., 챗봇: {chatbot_key}")
        
        # 챗봇 인스턴스 생성 또는 조회
        chatbot = get_or_create_chatbot(api_key, request_id)
        
        # 대화 길이 관리
        manage_conversation_length(chatbot)
        
        # 챗봇에 질문 전송
        response_content = chatbot.send_message(question, SYSTEM_PROMPT)
        
        logger.info(f"응답 생성 완료: {len(response_content)}자, 대화 길이: {len(chatbot.conversation_history)}")
        
        return {
            "text": response_content,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"질문 처리 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"질문 처리 중 오류가 발생했습니다: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """전역 예외 처리"""
    logger.error(f"예상치 못한 오류: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "서버 내부 오류가 발생했습니다"}
    )

if __name__ == "__main__":
    import uvicorn
    
    # 환경변수에서 설정 읽기
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 80))
    
    print(f"서버를 시작합니다: http://{host}:{port}")
    print("사용법:")
    if port == 80:
        print('  curl -X GET "http://49.50.132.254/agent?question=삼성전자%20주가" \\')
    else:
        print(f'  curl -X GET "http://49.50.132.254:{port}/agent?question=삼성전자%20주가" \\')
    print('    -H "Authorization: Bearer your-api-key" \\')
    print('    -H "X-NCP-CLOVASTUDIO-REQUEST-ID: your-request-id"')
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    ) 