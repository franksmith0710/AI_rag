"""
数据库模型定义
使用 SQLAlchemy ORM 定义所有数据库表结构
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from core.database import Base


class Tenant(Base):
    """
    租户表
    用于多租户隔离，每个租户拥有独立的用户、文档、会话
    """
    __tablename__ = "tenants"

    id = Column(Integer, primary_key=True, index=True)  # 主键
    name = Column(String(255), nullable=False)  # 租户名称
    created_at = Column(DateTime, server_default=func.now())  # 创建时间
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())  # 更新时间

    # 关联关系
    users = relationship("User", back_populates="tenant")  # 租户下的用户
    documents = relationship("Document", back_populates="tenant")  # 租户下的文档
    sessions = relationship("Session", back_populates="tenant")  # 租户下的会话


class User(Base):
    """
    用户表
    存储用户账号信息，支持 RBAC 角色
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False)  # 所属租户
    username = Column(String(100), unique=True, nullable=False, index=True)  # 用户名(唯一)
    password_hash = Column(String(255), nullable=False)  # 密码哈希
    role = Column(String(20), default="user")  # 角色: admin(管理员) / user(普通用户)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # 关联
    tenant = relationship("Tenant", back_populates="users")
    sessions = relationship("Session", back_populates="user")


class Document(Base):
    """
    文档表
    存储上传的文档元数据
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False)  # 所属租户
    title = Column(String(500), nullable=False)  # 文档标题
    file_name = Column(String(500))  # 原始文件名
    file_path = Column(String(1000))  # 文件存储路径
    file_size = Column(Integer)  # 文件大小(字节)
    file_type = Column(String(50))  # 文件类型 (pdf, docx, txt)
    status = Column(String(20), default="pending")  # 处理状态: pending(待处理) / completed(已完成)
    chunk_count = Column(Integer, default=0)  # 分块数量
    created_by = Column(Integer, ForeignKey("users.id"))  # 上传用户
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # 关联
    tenant = relationship("Tenant", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document")  # 文档的所有 chunks


class DocumentChunk(Base):
    """
    文档分块表
    存储文档分块后的文本内容，与向量库配合使用
    """
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)  # 所属文档
    chunk_index = Column(Integer, nullable=False)  # 分块序号
    text = Column(Text, nullable=False)  # 分块文本内容
    created_at = Column(DateTime, server_default=func.now())

    # 关联
    document = relationship("Document", back_populates="chunks")


class Session(Base):
    """
    会话表
    存储用户的对话会话，支持多轮对话
    """
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False)  # 所属租户
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)  # 所属用户
    title = Column(String(255))  # 会话标题
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # 关联
    tenant = relationship("Tenant", back_populates="sessions")
    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session")  # 会话的所有消息


class Message(Base):
    """
    消息表
    存储对话中的每条消息，包含用户问题和 AI 回答
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)  # 所属会话
    role = Column(String(20), nullable=False)  # 角色: user(用户) / assistant(AI)
    content = Column(Text, nullable=False)  # 消息内容
    sources = Column(JSON)  # 参考文档来源 (RAG 检索结果)
    created_at = Column(DateTime, server_default=func.now())

    # 关联
    session = relationship("Session", back_populates="messages")