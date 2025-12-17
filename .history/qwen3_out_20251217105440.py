import numpy as np
import torch
from server_wrapper_out import ServerMixin, host_model, send_request_vlm
import random
import os

from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor

def set_seed(seed: int):
    """
    固定随机种子，确保结果可复现。
    
    :param seed: 随机种子值
    """
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
        """
        初始化QWen3模型。

        :param model_name: 模型名称或路径
        :param device: 设备（cuda 或 cpu）
        """
        set_seed(seed)

        # 设置设备（默认使用 CUDA 或 CPU）
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # 加载模型和处理器
        print(f"Loading model {model_name}...")
        
        # 使用官方推荐的加载方式
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # 使用 bfloat16 精度
            device_map={"": self.device},  # 指定设备
            trust_remote_code=True,
        ).eval()
        
        # 使用 AutoProcessor 而不是 tokenizer
        self.processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        print(f"Model loaded on device: {next(self.model.parameters()).device}")
        print(f"Model max sequence length: {getattr(self.processor, 'model_max_length', 'N/A')}")
        
        # 生成配置
        self.generation_config = dict(
            max_new_tokens=512,
            do_sample=False,
            temperature=0,
            top_p=1.0,
            top_k=50,
        )

    def chat(self, txt: str) -> str:
        """
        纯文本对话接口
        
        :param txt: 输入文本
        :return: 模型响应
        """
        # 构造消息格式
        messages = [
            {
                "role": "system", 
                "content": "You are an AI assistant with advanced spatial reasoning capabilities. Your task is to choose the optimal option to find the target object."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": txt}
                ]
            }
        ]
        
        # 使用 processor 处理输入（官方推荐方式）
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # 将输入移到正确的设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 同步 CUDA
        if self.device.startswith('cuda'):
            torch.cuda.synchronize(self.device)
        
        # 生成文本
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                **self.generation_config
            )
        
        # 提取生成的部分（去掉输入部分）
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        
        # 解码输出
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return output_text[0] if output_text else ""

class Qwen3Client:
    """客户端类，用于向服务器发送请求"""
    
    def __init__(self, port: int = 8000):
        host = os.getenv("OUT_HOST", "localhost")
        self.url = f"http://{host}:{port}/v1/chat/completions"

    def chat(self, txt: str) -> str:
        """
        发送聊天请求
        
        :param txt: 输入文本
        :return: 模型响应
        """
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
    parser.add_argument("--test", action="store_true", help="Run a test query before starting server")
    args = parser.parse_args()

    print("Loading model...")

    class Qwen3Server(ServerMixin, QWen3):
        def process_payload(self, payload: dict) -> dict:
            # 处理 GET 请求（健康检查）
            if payload is None:  
                return {"status": "ok", "message": "Qwen3-VL service is running"}
                
            # 处理 POST 请求
            txt = payload.get("txt", "")
            if not txt:
                return {"error": "No text provided", "response": ""}
            
            try:
                response = self.chat(txt)
                return {"response": response}
            except Exception as e:
                print(f"Error processing request: {e}")
                import traceback
                traceback.print_exc()
                return {"error": str(e), "response": ""}

    qwen3 = Qwen3Server()
    print("Model loaded!")
    
    # 可选：运行测试
    if args.test:
        print("\nRunning test query...")
        test_response = qwen3.chat("Give me a short introduction to large language models.")
        print(f"Test response: {test_response}\n")
    
    print(f"Hosting on port {args.port}...")
    host_model(qwen3, name="qwen3", port=args.port)