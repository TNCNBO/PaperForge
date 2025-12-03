# Tavily 搜索服务器配置指南

## 关于 Tavily

Tavily 是专为 AI 应用和 LLM 设计的搜索引擎，提供高质量、相关性强的搜索结果。

## 特点

- **AI 优化**：专为 AI 应用设计的搜索结果
- **高质量**：经过过滤和优化的内容
- **快速响应**：支持基础和高级两种搜索模式
- **结构化输出**：易于 LLM 处理的格式

## 获取 API Key

1. 访问 [Tavily 官网](https://tavily.com)
2. 注册账号（支持 Google/GitHub 登录）
3. 在控制台获取 API Key
4. 免费套餐：1000 次/月搜索

## 配置步骤

### 1. 在 `mcp_agent.config.yaml` 中配置

已经配置好了 Tavily 服务器，只需填入 API Key：

```yaml
mcp:
  servers:
    tavily:
      args:
      - tools/tavily_search_server.py
      command: python
      description: AI-optimized search engine for high-quality, relevant results
      env:
        TAVILY_API_KEY: 'your_tavily_api_key_here'  # ← 在这里填入你的 API Key
        PYTHONPATH: .
```

### 2. 设置默认搜索服务器

```yaml
default_search_server: tavily
```

## 可用功能

### 1. tavily_search
标准搜索，返回多个结果：
- 支持基础/高级搜索模式
- 可指定结果数量（1-10）
- 可过滤特定域名

### 2. tavily_qna_search
问答模式，直接返回答案：
- 提供直接答案
- 附带来源链接
- 适合快速问答

## 使用示例

启动 DeepCode 后，AI 会自动使用 Tavily 进行 Web 搜索。你可以提问：

- "最新的 Python 3.12 有哪些新特性？"
- "搜索关于 Transformer 模型的最新研究"
- "查找 React 19 的性能优化方法"

## 备选方案

如果不想使用 Tavily，项目还支持：
- **Brave Search**（需要 Node.js MCP 服务器）
- **Bocha-MCP**（中文搜索引擎）

## 故障排除

### 错误：API key is not configured
确保在 `mcp_agent.config.yaml` 中正确设置了 `TAVILY_API_KEY`

### 错误：Connection timeout
- 检查网络连接
- Tavily 可能有速率限制，稍后重试

### 错误：Invalid API key
- 确认 API Key 复制正确
- 检查 API Key 是否已激活

## 更多信息

- [Tavily 文档](https://docs.tavily.com)
- [API 参考](https://docs.tavily.com/api-reference)
- [定价](https://tavily.com/pricing)
