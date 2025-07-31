import requests
import json
from typing import Optional, Dict, List, Any
import logging


class MiraeAIFestivalchatbot:
    def __init__(
        self,
        api_key: str,
        request_id: str,
        endpoint: str,
        logger: logging.Logger,
        tools,
    ):
        self.logger = logger
        self.logger.info("챗봇 초기화 시작")

        self.api_key = api_key
        self.request_id = request_id
        self.endpoint = endpoint
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self.request_id,
            "Content-Type": "application/json; charset=utf-8",
        }

        # 세부 API 설정 정보는 DEBUG 레벨로 출력
        self.logger.debug(
            f"API 설정: API Key 앞 10자리={self.api_key[:10]}..., Request ID={self.request_id}"
        )

        self.tools = tools

        # 사용 가능한 함수들
        self.available_functions = self.tools.available_functions
        
        # 대화 히스토리 저장
        self.conversation_history = []
        self.system_prompt = None

        self.logger.info("챗봇 초기화 완료")

    def clear_conversation(self):
        """대화 히스토리 초기화"""
        # 시스템 프롬프트와 최근 사용자 메시지 5개를 제외하고 초기화
        new_history = []
        new_history.append(self.conversation_history[0])
        
        new_history.extend(self.conversation_history[-6:])

        self.conversation_history = []
        self.system_prompt = None
        self.logger.info("대화 히스토리가 초기화되었습니다")

    def send_message(
        self, user_message: str, system_prompt: Optional[str] = None
    ) -> str:
        self.logger.info(f"메시지 처리 시작: {user_message[:50]}...")
        self.system_prompt = system_prompt
        
        # 시스템 프롬프트가 있지만 대화 히스토리가 비어있는 경우 (첫 메시지)
        if self.system_prompt and not self.conversation_history:
            self.conversation_history.append({"role": "system", "content": self.system_prompt})

        # 사용자 메시지 추가
        self.conversation_history.append({"role": "user", "content": user_message + " ToolCalls 활용하여 처리해줘. System_Prompt의 예시를 잘 활용해. 종목명으로 요청하면 임의로 티커코드로 변경하지 마"})

        request_data = {
            "messages": self.conversation_history.copy(),  # 복사본 사용
            "temperature": 0.2,
            "topP": 0.8,
            "topK": 0,
            "maxTokens": 2048,
            "repetitionPenalty": 0.9,
            "stop": [],
            "includeAiFilters": True,
            "seed": 0,
            "tools": self.tools.function_definitions,
            "tool_choice": "auto",
        }

        try:
            self.logger.info("API 요청 전송 중...")
            # tools 필드를 제거하여 요청 로그를 요약합니다.
            sanitized_request = {k: v for k, v in request_data.items() if k != "tools"}
            sanitized_request["tools_count"] = len(self.tools.function_definitions)
            sanitized_request["conversation_length"] = len(self.conversation_history)
            self.logger.debug(
                f"API 요청 데이터(요약): {json.dumps(sanitized_request, ensure_ascii=False, indent=2)}"
            )
            response = requests.post(
                self.endpoint, headers=self.headers, json=request_data
            )

            return self._handle_response(response, request_data, limit_count=1)

        except Exception as e:
            error_msg = f"예상치 못한 오류가 발생했습니다: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return error_msg

    def _handle_function_calls(
        self,
        response_message: Dict[str, Any],
        request_data: Dict[str, Any],
        limit_count: int,
    ) -> str:
        self.logger.info("Function Calling 처리 시작")

        # 대화 히스토리에 AI 응답 추가
        # self.conversation_history.append(response_message)

        # toolCalls 또는 toolCalls 키 확인
        toolCalls = response_message.get("toolCalls")

        if not toolCalls:
            self.logger.error("❌ toolCalls를 찾을 수 없습니다")
            return "Function calling 정보를 찾을 수 없습니다."

        for i, tool_call in enumerate(toolCalls):
            function_name = tool_call["function"]["name"]
            function_args = tool_call["function"]["arguments"]

            self.logger.info(f"함수 호출 #{i+1}: {function_name}")
            self.logger.debug(f"   Arguments: {function_args}")
            self.logger.debug(f"   Tool Call ID: {tool_call['id']}")

            if function_name in self.available_functions:
                try:
                    self.logger.debug("   함수 실행 중...")

                    # 함수 호출 - **function_args로 키워드 인수 전달
                    function_result = self.available_functions[function_name](
                        **function_args
                    )
                    self.logger.info("   함수 실행 성공")
                    self.logger.debug(
                        f"   결과: {json.dumps(function_result, ensure_ascii=False, indent=2)}"
                    )

                    tool_message = {
                        "role": "tool",
                        "toolCallId": tool_call["id"],
                        "content": json.dumps(function_result, ensure_ascii=False),
                    }
                    # self.conversation_history.append(tool_message)
                    request_data["messages"].append(tool_message)
                    self.logger.debug("   도구 응답 메시지 추가")

                except TypeError as e:
                    # 함수 매개변수 불일치 에러
                    error_msg = f"함수 매개변수 오류: {str(e)}"
                    self.logger.error(f"   ❌ {error_msg}")
                    self.logger.error(f"   - 함수명: {function_name}")
                    self.logger.error(f"   - 전달된 인수: {function_args}")

                    tool_message = {
                        "role": "tool",
                        "toolCallId": tool_call["id"],
                        "content": json.dumps(
                            {
                                "success": False,
                                "error": str(e),
                                "message": error_msg,
                            },
                            ensure_ascii=False,
                        ),
                    }
                    request_data["messages"].append(tool_message)
                    # self.conversation_history.append(tool_message)

                except Exception as e:
                    error_msg = f"함수 실행 중 오류: {str(e)}"
                    self.logger.error(f"   ❌ {error_msg}")

                    tool_message = {
                        "role": "tool",
                        "toolCallId": tool_call["id"],
                        "content": json.dumps(
                            {"success": False, "error": str(e), "message": error_msg},
                            ensure_ascii=False,
                        ),
                    }
                    request_data["messages"].append(tool_message)
                    # self.conversation_history.append(tool_message)
            else:
                error_msg = f"지원하지 않는 함수입니다: {function_name}"
                self.logger.error(f"   ❌ {error_msg}")

                tool_message = {
                    "role": "tool",
                    "toolCallId": tool_call["id"],
                    "content": json.dumps(
                        {"success": False, "message": error_msg}, ensure_ascii=False
                    ),
                }
                request_data["messages"].append(tool_message)
                # self.conversation_history.append(tool_message)

        # 최종 응답 요청
        # request_data["messages"] = self.conversation_history.copy()

        self.logger.debug("최종 응답 요청 준비")
        self.logger.debug(f"   총 메시지 수: {len(self.conversation_history)}")
        self.logger.debug(f"   총 메시지 수: {len(request_data['messages'])}")

        try:
            self.logger.info(f"🌐 최종 응답 요청 전송...")

            response = requests.post(
                self.endpoint, headers=self.headers, json=request_data
            )
            return self._handle_response(response, request_data, limit_count=limit_count)

        except Exception as e:
            error_msg = f"함수 호출 후 응답 생성 중 오류가 발생했습니다: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return error_msg

    def _handle_response(
        self,
        response: requests.Response,
        request_data: Dict[str, Any],
        limit_count: int,
    ) -> str:

        self.logger.info(f"API 응답 수신: HTTP {response.status_code}")
        if response.status_code != 200:
            self.logger.error(f"❌ HTTP 에러: {response.status_code}")
            self.logger.error(f"응답 내용: {response.text}")
            return f"HTTP 에러: {response.status_code} - {response.text}"

        result = response.json()
        self.logger.debug("응답 파싱 완료")

        if result.get("status", {}).get("code") == "20000":
            response_message = result["result"]["message"]
            self.logger.info("성공 응답 수신")
            self.logger.debug(result)

            # Function Calling 키 이름 확인 (toolCalls 또는 toolCalls)
            toolCalls = response_message.get("toolCalls")

            if toolCalls:
                if limit_count > 5:
                    return "죄송합니다. 사용자의 요청을 처리할 수 없습니다. 다시 시도해주세요."
                self.logger.info(f"Function Calling 감지: {len(toolCalls)}개 호출")
                return self._handle_function_calls(response_message, request_data, limit_count + 1)
            else:
                content = response_message.get("content", "")
                # content에 도구 실행 요청이 포함되어 있는지 확인
                if any(keyword in content for keyword in ["데이터를 확인하겠습니다", "도구를 실행하여", "도구를 호출하여", "도구를 활용하여", "데이터를 제공하겠습니다", "쿼리를 실행하여"]):
                    self.logger.info("도구 실행 요청이 감지되어 다시 처리합니다.")
                    self.conversation_history.append({"role": "user", "content": "그래 도구를 활용하여 대답해줘"})
                    
                    # 새로운 요청 데이터 생성
                    new_request_data = {
                        "messages": self.conversation_history.copy(),
                        "temperature": 0.2,
                        "topP": 0.8,
                        "topK": 0,
                        "maxTokens": 2048,
                        "repetitionPenalty": 0.9,
                        "stop": [],
                        "includeAiFilters": True,
                        "seed": 0,
                        "tools": self.tools.function_definitions,
                        "tool_choice": "auto",
                    }
                    
                    try:
                        self.logger.info("도구 실행을 위한 재요청 전송...")
                        new_response = requests.post(
                            self.endpoint, headers=self.headers, json=new_request_data
                        )
                        return self._handle_response(new_response, new_request_data, limit_count)
                    except Exception as e:
                        self.logger.error(f"재요청 중 오류: {str(e)}")
                        return content  # 원래 응답 반환
                
                # AI 응답을 대화 히스토리에 추가
                self.conversation_history.append({"role": "assistant", "content": ""})
                self.logger.debug(f"일반 응답: {content}")
                self.logger.debug(f"현재 대화 길이: {len(self.conversation_history)}")
                return content
        else:
            error_msg = f"API 오류: {result.get('status', {}).get('message', '알 수 없는 오류')}"
            self.logger.error(f"❌ {error_msg}")
            return error_msg
