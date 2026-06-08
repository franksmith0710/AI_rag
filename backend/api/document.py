"""
文档管理 API 路由模块
提供文档上传、列表、详情、删除、处理等接口
支持 PDF/Word/TXT 格式文档
"""
import os
import uuid
import logging
import asyncio
from typing import Optional
import aiofiles
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from core.logging_config import setup_logging
from models.schemas import DocumentResponse, DocumentListResponse, DocumentChunkListResponse
from services import doc_service
from api.auth import get_current_user
from models.schemas import UserResponse
from utils.common import ApiResponse
from core.config import get_settings

logger = setup_logging("api.document")
router = APIRouter(prefix="/documents", tags=["文档"])
settings = get_settings()


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    is_global: bool = Form(False),
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    上传文档接口

    支持格式: .pdf, .docx, .doc, .txt, .md, .jpg, .jpeg, .png, .bmp, .tiff
    is_global=True 时上传到全局共享租户 (仅 admin 可用)

    Args:
        file: 上传的文件
        is_global: 是否上传到全局共享
        current_user: 当前登录用户
        db: 数据库会话

    Returns:
        文档信息

    Raises:
        HTTPException: 不支持的文件类型 或 权限不足
    """
    # 检查文件类型
    file_ext = os.path.splitext(file.filename)[1].lower()
    allowed_exts = [".pdf", ".docx", ".doc", ".txt", ".md", ".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    if file_ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件类型: {file_ext}"
        )

    # 全局共享校验：只有 admin 可以上传到全局租户
    if is_global and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有管理员可以上传到全局共享"
        )

    # 确定租户 ID
    tenant_id = 0 if is_global else current_user.tenant_id

    # 生成唯一文件名，防止冲突
    file_id = str(uuid.uuid4())
    file_name = f"{file_id}{file_ext}"
    file_path = os.path.join(settings.upload_dir, file_name)

    # 确保目录存在
    os.makedirs(settings.upload_dir, exist_ok=True)

    # 保存文件到本地
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    # 创建文档记录
    doc = await doc_service.create_document(
        db=db,
        tenant_id=tenant_id,
        title=file.filename.replace(file_ext, ""),
        file_name=file.filename,
        file_path=file_path,
        file_size=len(content),
        file_type=file_ext[1:],
        user_id=current_user.id
    )
    await db.commit()

    return DocumentResponse.model_validate(doc)


@router.post("/process/{document_id}")
async def process_document(
    document_id: int,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    处理文档接口
    解析文档文本，进行分块和向量化存储

    Args:
        document_id: 文档 ID
        current_user: 当前登录用户
        db: 数据库会话

    Returns:
        处理结果

    Raises:
        HTTPException: 文档不存在
    """
    # 获取文档
    doc = await doc_service.get_document_by_id(db, document_id, current_user.tenant_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在"
        )

    # 已经处理过则跳过
    if doc.status == "completed":
        return ApiResponse.success(message="文档已处理完成")

    logger.info(f"开始处理文档 document_id={document_id}, file={doc.file_name}")
    try:
        # 提取文本
        text_content = await asyncio.to_thread(doc_service.extract_text_from_file, doc.file_path, doc.file_type)

        # 处理文档 (分块 + 向量化)
        success = await doc_service.process_document(db, document_id, text_content, current_user.tenant_id)
        await db.commit()

        if success:
            logger.info(f"文档 {document_id} 处理成功")
            return ApiResponse.success(message="文档处理成功")
        else:
            logger.error(f"文档 {document_id} 处理失败（空文本）")
            return JSONResponse(
                status_code=500,
                content=ApiResponse.error(message="文档处理失败（空文本）", code=500)
            )

    except Exception as e:
        logger.error(f"文档 {document_id} 处理异常: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ApiResponse.error(message=f"处理出错: {str(e)}", code=500)
        )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    skip: int = 0,
    limit: int = 20,
    is_global: Optional[bool] = Query(None, description="True=仅全局, False=仅个人, None=全部"),
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    获取文档列表（支持全局共享过滤）

    Args:
        skip: 跳过条数
        limit: 返回条数
        is_global: True=仅全局共享, False=仅个人, None=全部
        current_user: 当前登录用户
        db: 数据库会话

    Returns:
        文档列表和总数
    """
    only_global = is_global is True
    return await doc_service.get_documents(db, current_user.tenant_id, skip, limit, include_global=True, only_global=only_global)


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    获取文档详情（支持查看全局共享文档）

    Args:
        document_id: 文档 ID
        current_user: 当前登录用户
        db: 数据库会话

    Returns:
        文档详情
    """
    # 支持查看全局文档：不限制 tenant_id，直接查询
    from sqlalchemy import select
    from models.db_models import Document
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在"
        )

    # 非全局文档，需要校验租户权限
    if doc.tenant_id != 0 and doc.tenant_id != current_user.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权查看该文档"
        )

    return DocumentResponse.model_validate(doc)


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    删除文档

    删除文档记录、分块数据和向量数据

    Args:
        document_id: 文档 ID
        current_user: 当前登录用户
        db: 数据库会话

    Returns:
        删除结果
    """
    # 获取文档信息（不限制 tenant_id，因为可能删除全局文档）
    from sqlalchemy import select
    from models.db_models import Document
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在"
        )

    # 全局租户文档权限校验：只有 admin 可以删除
    if doc.tenant_id == 0 and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有管理员可以删除全局共享文档"
        )

    # 非全局文档，只能删除自己租户的
    if doc.tenant_id != 0 and doc.tenant_id != current_user.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权删除其他租户的文档"
        )

    # 执行删除
    success = await doc_service.delete_document(db, document_id, doc.tenant_id)
    await db.commit()

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除失败"
        )

    return ApiResponse.success(message="删除成功")


@router.get("/{document_id}/chunks", response_model=DocumentChunkListResponse)
async def get_document_chunks(
    document_id: int,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    获取文档的所有 chunks

    权限逻辑：
    - 全局文档(tenant_id=0)：所有用户可查看
    - 租户文档：仅该租户用户可查看
    """
    from sqlalchemy import select
    from models.db_models import Document, DocumentChunk

    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在"
        )

    # 权限校验：tenant_id=0 或 tenant_id=当前用户.tenant_id
    if doc.tenant_id != 0 and doc.tenant_id != current_user.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权查看该文档的 chunks"
        )

    # 查询所有 chunks，按 chunk_index 排序
    result = await db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
    )
    chunks = result.scalars().all()

    items = [
        {"chunk_index": c.chunk_index, "text": c.text}
        for c in chunks
    ]

    return DocumentChunkListResponse(total=len(items), items=items)