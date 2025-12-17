import numpy as np
import torch
from server_wrapper_out import ServerMixin, host_model, send_request_vlm
import random
import os

from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
os['CUDA_VISIBLE_DEVICES']=1
def set_seed(seed: int):
    """固定随机种子，确保结果可复现。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class QWen3:
    """QWen3-VL model using official API."""

    def __init__(self, model_name: str = "model/Qwen3-VL-8B-Instruct", device: str = None, seed: int = 2025) -> None: 
        set_seed(seed)

        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = device

        print(f"Loading model {model_name}...")
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map={"": self.device},
            trust_remote_code=True,
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        print(f"Model loaded on device: {next(self.model.parameters()).device}")
        
        self.generation_config = dict(
            max_new_tokens=512,
            do_sample=False,
            temperature=0,
            top_p=1.0,
            top_k=50,
        )

    def _normalize_messages(self, messages):
        """
        规范化消息格式，确保 content 始终是列表格式
        
        输入可能是：
        1. [{"role": "user", "content": "text"}]  # 字符串格式
        2. [{"role": "user", "content": [{"type": "text", "text": "..."}]}]  # 标准格式
        3. [{"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image_url", ...}]}]  # 多模态
        """
        normalized = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # 如果 content 是字符串，转换为列表格式
            if isinstance(content, str):
                normalized_content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                normalized_content = []
                for item in content:
                    if isinstance(item, dict):
                        # 处理 image_url 格式（来自 agents.py）
                        if item.get("type") == "image_url":
                            image_url = item.get("image_url", {})
                            if isinstance(image_url, dict):
                                url = image_url.get("url", "")
                            else:
                                url = str(image_url)
                            
                            # Qwen3-VL 期望 "image" 类型
                            normalized_content.append({
                                "type": "image",
                                "image": url  # data:image/jpeg;base64,...
                            })
                        # 处理标准格式
                        elif item.get("type") in ["text", "image", "video"]:
                            normalized_content.append(item)
                        else:
                            # 未知格式，尝试作为文本
                            if "text" in item:
                                normalized_content.append({"type": "text", "text": item["text"]})
                    elif isinstance(item, str):
                        # 字符串直接作为文本
                        normalized_content.append({"type": "text", "text": item})
            else:
                # 其他类型转为字符串
                normalized_content = [{"type": "text", "text": str(content)}]
            
            normalized.append({
                "role": role,
                "content": normalized_content
            })
        
        return normalized

    def chat(self, txt: str = None, messages: list = None) -> str:
        """
        对话接口，支持两种输入：
        1. 纯文本：txt="..."
        2. 消息列表：messages=[{"role": "user", "content": ...}]
        """
        if messages is None:
            if txt is None:
                return ""
            
            # 纯文本模式
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": txt}]
                }
            ]
        else:
            # 规范化消息格式
            messages = self._normalize_messages(messages)
        
        try:
            # 使用 processor 处理输入
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # 移到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            if self.device.startswith('cuda'):
                torch.cuda.synchronize(self.device)
            
            # 生成
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    **self.generation_config
                )
            
            # 提取生成部分
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            
            # 解码
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            return output_text[0] if output_text else ""
        
        except Exception as e:
            print(f"Error in chat: {e}")
            import traceback
            traceback.print_exc()
            return ""

class Qwen3Client:
    """客户端类"""
    
    def __init__(self, port: int = 8000):
        host = os.getenv("OUT_HOST", "localhost")
        self.url = f"http://{host}:{port}/v1/chat/completions"

    def chat(self, txt: str) -> str:
        try:
            response = send_request_vlm(self.url, timeout=20, txt=txt)
            return response["response"]
        except Exception as e:
            print(f"Request failed: {e}")
            return "-1"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--test",
        type=bool,  # 指定参数类型为布尔值
        default=False,  # 可选：设置默认值（根据需求调整）
        help="Run a test query (specify True/False, default: False)"
    )
    args = parser.parse_args()

    print("Loading model...")

    class Qwen3Server(ServerMixin, QWen3):
        def process_payload(self, payload: dict) -> dict:
            # 健康检查
            if payload is None:  
                return {"status": "ok", "message": "Qwen3-VL service is running"}
            
            try:
                # 支持两种请求格式
                
                # 格式1：纯文本 {"txt": "..."}
                if "txt" in payload:
                    txt = payload.get("txt", "")
                    if not txt:
                        return {"error": "Empty text", "response": ""}
                    response = self.chat(txt=txt)
                    return {"response": response}
                
                # 格式2：OpenAI 格式 {"messages": [...]}
                elif "messages" in payload:
                    messages = payload.get("messages", [])
                    if not messages:
                        return {"error": "Empty messages", "response": ""}
                    response = self.chat(messages=messages)
                    
                    # 返回 OpenAI 兼容格式
                    return {
                        "choices": [{
                            "message": {
                                "content": response,
                                "role": "assistant"
                            }
                        }]
                    }
                
                else:
                    return {"error": "Invalid payload format", "response": ""}
                    
            except Exception as e:
                print(f"Error processing request: {e}")
                import traceback
                traceback.print_exc()
                return {"error": str(e), "response": ""}

    qwen3 = Qwen3Server()
    print("Model loaded!")
    
    if args.test:
        print("\n=== Running tests ===")
        
        # 测试1：纯文本
        print("\n1. Testing pure text format...")
        response1 = qwen3.process_payload({"txt": "What is 2+2?"})
        print(f"Response: {response1}")
        
        # 测试2：OpenAI 格式（字符串 content）
        print("\n2. Testing OpenAI format (string content)...")
        response2 = qwen3.process_payload({
            "messages": [
                {"role": "user", "content": "What is the capital of France?"}
            ]
        })
        print(f"Response: {response2}")
        
        # 测试3：OpenAI 格式（列表 content）
        print("\n3. Testing OpenAI format (list content)...")
        response3 = qwen3.process_payload({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe AI in one sentence."}
                    ]
                }
            ]
        })
        print(f"Response: {response3}")
        print("\n=== Tests complete ===\n")
    
    print(f"Hosting on port {args.port}...")
    host_model(qwen3, name="qwen3", port=args.port)