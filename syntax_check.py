#!/usr/bin/env python3
"""
语法检查脚本 - 验证修复后的 audio_generator.py 是否有语法错误
"""

import ast
import sys
from pathlib import Path

def check_syntax(file_path):
    """检查 Python 文件的语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # 尝试解析 AST
        ast.parse(source_code)
        print(f"✅ {file_path}: 语法检查通过")
        return True
        
    except SyntaxError as e:
        print(f"❌ {file_path}: 语法错误")
        print(f"   行 {e.lineno}: {e.text.strip() if e.text else ''}")
        print(f"   错误: {e.msg}")
        return False
        
    except Exception as e:
        print(f"❌ {file_path}: 检查失败 - {e}")
        return False

def main():
    """主函数"""
    files_to_check = [
        "news_podcast/audio_generator.py",
        "demo/inference_from_file.py"  # 作为参考
    ]
    
    all_passed = True
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            if not check_syntax(file_path):
                all_passed = False
        else:
            print(f"⚠️  文件不存在: {file_path}")
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有文件语法检查通过！")
        
        # 额外检查：验证关键修复点
        print("\n检查关键修复点...")
        with open("news_podcast/audio_generator.py", 'r') as f:
            content = f.read()
            
        checks = [
            ("text=[text],  # Wrap in list for batch processing", "处理器调用格式"),
            ("outputs.speech_outputs[0]", "音频输出获取"),
            ("self.processor.save_audio", "音频保存方法"),
            ("torch_dtype=torch.bfloat16", "模型数据类型"),
            ("device_map='cuda'", "设备映射"),
            ("sample_rate = 24000", "采样率设置")
        ]
        
        for check_str, description in checks:
            if check_str in content:
                print(f"✅ {description}: 已修复")
            else:
                print(f"❌ {description}: 可能未正确修复")
                all_passed = False
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())