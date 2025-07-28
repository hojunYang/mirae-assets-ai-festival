import os
from datetime import datetime
import logging
from chatbot import MiraeAIFestivalchatbot
from tools import Tools

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
            f"{LOG_DIR}/chatbot_{datetime.now().strftime(DATE_FORMAT)}.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = open("clova.prompt", "r", encoding="utf-8").read()
CLOVASTUDIO_ENDPOINT = (
    "https://clovastudio.stream.ntruss.com/testapp/v3/chat-completions/HCX-005"
)


def main():
    logger.info("미래에셋 AI 챗봇 시작")

    print("=" * 60)
    print("CLOVA Studio 챗봇에 오신 것을 환영합니다!")
    print("실시간 주가 조회 기능이 포함되어 있습니다!")
    print("로그는 chatbot.log 파일에서 확인하실 수 있습니다!")
    print("=" * 60)

    # API 키 설정
    api_key = os.getenv("CLOVASTUDIO_API_KEY")
    request_id = os.getenv("CLOVASTUDIO_REQUEST_ID")
    endpoint = CLOVASTUDIO_ENDPOINT

    tools = Tools(logger)

    if not api_key:
        logger.warning("CLOVA Studio API 키가 환경변수에 설정되지 않음")
        print("CLOVA Studio API 키를 설정해주세요.")
        print("방법 1: 환경변수 설정 - export CLOVASTUDIO_API_KEY='your-api-key'")
        print("방법 2: 직접 입력")
        api_key = input("API 키를 입력하세요: ").strip()

        if not api_key:
            logger.error("API 키 미제공으로 프로그램 종료")
            print("API 키가 제공되지 않았습니다. 프로그램을 종료합니다.")
            return

    try:
        # 챗봇 인스턴스 생성
        logger.info("챗봇 인스턴스 생성")
        chatbot = MiraeAIFestivalchatbot(api_key, request_id, endpoint, logger, tools)

        # 대화 시작
        print("\n대화를 시작해보세요! (종료하려면 'quit' 또는 '종료'를 입력하세요)")
        print("=" * 60)

        while True:
            user_input = input("\n사용자: ").strip()

            if user_input.lower() in ["quit", "종료", "exit", "나가기"]:
                logger.info("사용자 요청으로 프로그램 종료")
                print("\n이용해 주셔서 감사합니다. 좋은 하루 되세요!")
                break

            if not user_input:
                logger.debug("빈 메시지 입력됨")
                print("메시지를 입력해주세요.")
                continue

            print("응답을 기다리는 중...")
            logger.info(f"사용자 입력: {user_input}")

            response = chatbot.send_message(user_input, SYSTEM_PROMPT)
            print(f"AI: {response}")

    except Exception as e:
        logger.critical(f"치명적 오류 발생: {str(e)}")
        print(f"오류가 발생했습니다: {str(e)}")


if __name__ == "__main__":
    main()
