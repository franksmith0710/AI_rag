"""
认证 API 路由模块
提供用户注册、登录、获取当前用户信息等接口
使用 JWT Token 进行身份验证
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from models.schemas import UserCreate, UserLogin, TokenResponse, UserResponse
from services import auth_service

router = APIRouter(prefix="/auth", tags=["认证"])
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """
    获取当前用户信息
    通过 JWT Token 验证用户身份
    用于需要登录才能访问的接口的依赖注入

    Args:
        credentials: HTTP Bearer Token
        db: 数据库会话

    Returns:
        当前用户信息

    Raises:
        HTTPException: Token 无效或用户不存在
    """
    token = credentials.credentials

    # 解码 Token
    payload = auth_service.decode_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭据"
        )

    # 获取用户 ID
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的token"
        )

    # 查询用户
    try:
        user = await auth_service.get_user_by_id(db, int(user_id))
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的token"
        )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在"
        )

    return UserResponse(
        id=user.id,
        tenant_id=user.tenant_id,
        username=user.username,
        role=user.role
    )


@router.post("/register", response_model=TokenResponse)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    用户注册接口

    Args:
        user_data: 用户注册信息 (tenant_id, username, password, role)
        db: 数据库会话

    Returns:
        JWT Token 和用户信息

    Raises:
        HTTPException: 用户名已存在
    """
    # 检查用户名是否已存在
    existing_user = await auth_service.get_user_by_username(db, user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户名已存在"
        )

    # 根据角色设置租户
    # 所有用户（包括 admin）都创建独立私有租户
    # admin 通过 role 权限访问全局共享文档 (tenant_id=0)
    if user_data.role != "admin":
        user_data.role = "user"
    tenant = await auth_service.create_tenant(db, f"租户_{user_data.username}")
    user_data.tenant_id = tenant.id

    # 创建用户
    user = await auth_service.create_user(db, user_data)
    await db.commit()

    # 生成 Token
    access_token = auth_service.create_access_token(
        data={"sub": str(user.id), "tenant_id": str(user.tenant_id), "role": user.role}
    )

    return TokenResponse(
        access_token=access_token,
        user=UserResponse(
            id=user.id,
            tenant_id=user.tenant_id,
            username=user.username,
            role=user.role
        )
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    login_data: UserLogin,
    db: AsyncSession = Depends(get_db)
):
    """
    用户登录接口

    Args:
        login_data: 用户名和密码
        db: 数据库会话

    Returns:
        JWT Token 和用户信息

    Raises:
        HTTPException: 用户名或密码错误
    """
    # 验证用户
    user = await auth_service.authenticate_user(db, login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误"
        )

    # 生成 Token
    access_token = auth_service.create_access_token(
        data={"sub": str(user.id), "tenant_id": str(user.tenant_id), "role": user.role}
    )

    return TokenResponse(
        access_token=access_token,
        user=UserResponse(
            id=user.id,
            tenant_id=user.tenant_id,
            username=user.username,
            role=user.role
        )
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: UserResponse = Depends(get_current_user)
):
    """
    获取当前登录用户信息

    需要 Bearer Token 认证

    Args:
        current_user: 通过依赖注入获取的当前用户

    Returns:
        当前用户信息
    """
    return current_user