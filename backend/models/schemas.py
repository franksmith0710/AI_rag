"""
Pydantic 数据模型
用于 API 请求参数校验和响应数据格式化
与 db_models 配合使用，db_models 用于数据库，schemas 用于 API
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ==================== 租户相关 ====================

class TenantCreate(BaseModel):
    """创建租户请求"""
    name: str  # 租户名称


class TenantResponse(BaseModel):
    """租户响应"""
    id: int
    name: str
    created_at: datetime

    class Config:
        from_attributes = True  # 允许从 ORM 对象创建


# ==================== 用户相关 ====================

class UserCreate(BaseModel):
    """创建用户请求"""
    username: str  # 用户名
    password: str  # 密码
    role: str = "user"  # 角色：admin / user，默认普通用户
    tenant_id: int = None  # 可选，管理员可指定租户 ID


class UserLogin(BaseModel):
    """用户登录请求"""
    username: str
    password: str


class UserResponse(BaseModel):
    """用户响应 (不含敏感信息)"""
    id: int
    tenant_id: int
    username: str
    role: str

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """登录响应"""
    access_token: str  # JWT Token
    token_type: str = "bearer"
    user: UserResponse  # 用户信息


# ==================== 文档相关 ====================

class DocumentCreate(BaseModel):
    """创建文档请求 (内部使用)"""
    title: str
    file_name: str
    file_size: int
    file_type: str
    file_path: str


class DocumentResponse(BaseModel):
    """文档响应"""
    id: int
    tenant_id: int
    title: str
    file_name: Optional[str]
    file_size: Optional[int]
    file_type: Optional[str]
    status: str  # pending / completed
    chunk_count: int
    created_by: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    total: int  # 总数
    items: List[DocumentResponse]  # 文档列表


class DocumentStatusUpdate(BaseModel):
    """更新文档状态请求"""
    status: str
    chunk_count: Optional[int] = 0


class DocumentChunkResponse(BaseModel):
    """文档分块响应"""
    chunk_index: int
    text: str

    class Config:
        from_attributes = True


class DocumentChunkListResponse(BaseModel):
    """文档分块列表响应"""
    total: int
    items: List[DocumentChunkResponse]


# ==================== 消息相关 ====================

class MessageResponse(BaseModel):
    """消息响应"""
    id: int
    session_id: int
    role: str  # user / assistant
    content: str
    sources: Optional[List[Dict[str, Any]]]  # 参考文档
    created_at: datetime

    class Config:
        from_attributes = True


# ==================== 会话相关 ====================

class SessionCreate(BaseModel):
    """创建会话请求"""
    title: Optional[str] = None  # 可选的会话标题


class SessionResponse(BaseModel):
    """会话响应"""
    id: int
    tenant_id: int
    user_id: int
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse] = []  # 会话消息列表

    class Config:
        from_attributes = True


class SessionListResponse(BaseModel):
    """会话列表响应"""
    total: int
    items: List[SessionResponse]


# ==================== 聊天相关 ====================

class ChatRequest(BaseModel):
    """聊天请求"""
    session_id: int  # 会话 ID
    message: str  # 用户消息


class ChatResponse(BaseModel):
    """聊天响应"""
    session_id: int
    message: str  # AI 回答
    sources: Optional[List[Dict[str, Any]]]  # 参考文档


class StreamChatRequest(BaseModel):
    """流式聊天请求 (预留)"""
    session_id: int
    message: str


# ==================== 批量上传相关 ====================

class BatchUploadResult(BaseModel):
    """单个文件上传结果"""
    file_name: str
    success: bool
    document_id: Optional[int] = None
    error: Optional[str] = None


class BatchUploadResponse(BaseModel):
    """批量上传响应"""
    total: int
    success: int
    failed: int
    results: List[BatchUploadResult]


# ==================== 批量处理相关 ====================

class BatchProcessRequest(BaseModel):
    """批量处理请求"""
    document_ids: List[int]


class BatchProcessResult(BaseModel):
    """单个文档处理结果"""
    document_id: int
    success: bool
    error: Optional[str] = None


class BatchProcessResponse(BaseModel):
    """批量处理响应"""
    total: int
    success: int
    failed: int
    results: List[BatchProcessResult]