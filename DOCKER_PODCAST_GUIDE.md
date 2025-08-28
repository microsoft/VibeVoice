# VibeVoice新闻播客Docker使用指南

## 🐳 快速开始

### 1. 构建并启动容器

```bash
cd /path/to/VibeVoice/docker
docker compose up --build -d
```

### 2. 进入容器

```bash
docker compose exec vibevoice bash
```

### 3. 运行容器设置检查

```bash
./setup_container.sh
```

## 📻 生成新闻播客

### 基础用法

```bash
# 测试管道（仅生成文本，不生成音频）
python test_news_pipeline.py

# 生成完整播客（包含音频）
python generate_news_podcast.py
```

### 自定义参数

```bash
# 使用3个说话人
python generate_news_podcast.py --speakers 3

# 使用更大的模型
python generate_news_podcast.py --model-path WestZhang/VibeVoice-Large-pt

# 自定义输出目录
python generate_news_podcast.py --output-dir /app/podcast_output

# 获取更多新闻
python generate_news_podcast.py --news-limit 20 --max-news-items 15

# 查看可用语音
python generate_news_podcast.py --list-voices
```

### 完整参数示例

```bash
python generate_news_podcast.py \
    --model-path microsoft/VibeVoice-1.5B \
    --speakers 4 \
    --news-limit 25 \
    --max-news-items 12 \
    --output-dir /app/podcast_output
```

## 📁 文件访问

生成的播客文件可以通过以下方式访问：

### 从容器内部
```bash
ls /app/podcast_output/
```

### 从宿主机
```bash
ls ./docker/podcast_output/
```

### 生成的文件格式
- `news_podcast_YYYYMMDD_HHMMSS.wav` - 最终音频文件
- `news_podcast_YYYYMMDD_HHMMSS_dialogue.txt` - 对话脚本
- `news_podcast_YYYYMMDD_HHMMSS_news.json` - 原始新闻数据

## 🔧 环境配置

容器预设了以下环境变量：

```bash
MODEL_PATH=microsoft/VibeVoice-1.5B
OLLAMA_URL=http://172.36.237.245:11434
OLLAMA_MODEL=qwen2.5-coder:1.5b
```

### 修改配置

可以在`docker-compose.yml`中修改这些设置：

```yaml
environment:
  - MODEL_PATH=WestZhang/VibeVoice-Large-pt
  - OLLAMA_URL=http://your-ollama-server:11434
  - OLLAMA_MODEL=llama3.1
```

## 🐛 故障排除

### 1. Ollama连接问题

```bash
# 检查Ollama连接
curl http://172.36.237.245:11434/api/version

# 在容器内测试连接
python -c "
from news_podcast.ollama_processor import OllamaProcessor
processor = OllamaProcessor()
print('✓ Connected' if processor.check_ollama_connection() else '✗ Failed')
"
```

### 2. GPU不可用

```bash
# 检查GPU状态
nvidia-smi

# 在容器内检查
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

### 3. 内存不足

- 使用较小的模型：`microsoft/VibeVoice-1.5B`
- 减少说话人数量：`--speakers 2`
- 减少新闻数量：`--news-limit 10`

### 4. 音频生成失败

```bash
# 检查soundfile安装
python -c "import soundfile; print('✓ soundfile available')"

# 检查语音文件
ls /app/demo/voices/en-*.wav
```

## 📊 性能优化

### GPU内存优化
```bash
# 使用1.5B模型而不是7B
python generate_news_podcast.py --model-path microsoft/VibeVoice-1.5B

# 减少并行处理
python generate_news_podcast.py --speakers 2
```

### 网络优化
```bash
# 减少新闻获取数量
python generate_news_podcast.py --news-limit 10 --max-news-items 6
```

## 🔄 容器管理

```bash
# 查看容器状态
docker compose ps

# 查看日志
docker compose logs -f

# 重启容器
docker compose restart

# 停止容器
docker compose down

# 重新构建
docker compose up --build
```

## 🎯 示例工作流程

```bash
# 1. 启动容器
cd docker && docker compose up -d

# 2. 进入容器
docker compose exec vibevoice bash

# 3. 运行设置检查
./setup_container.sh

# 4. 快速测试
python test_news_pipeline.py

# 5. 生成播客
python generate_news_podcast.py --speakers 3

# 6. 查看生成的文件
ls /app/podcast_output/

# 7. 从宿主机访问文件
exit
ls ./podcast_output/
```

这样你就可以在容器中完整地运行新闻播客生成功能了！