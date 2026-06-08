"""
数据库初始化脚本
用于首次启动时创建数据库表和默认数据
运行方式: python init_db.py
"""
import asyncio
import sys
import os
import logging

# 将 backend 目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.database import async_engine, Base, async_session_maker
from models.db_models import Tenant, User
from services.auth_service import get_password_hash

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


async def init_default_data():
    """
    初始化默认数据
    - 创建全局共享租户 (tenant_id=0)
    - 创建默认租户 (如果不存在)
    - 创建多个用户账号
    """
    async with async_session_maker() as session:
        from sqlalchemy import select

        # 1. 确保全局共享租户存在 (tenant_id=0)
        global_tenant_result = await session.execute(
            select(Tenant).where(Tenant.id == 0)
        )
        global_tenant = global_tenant_result.scalar_one_or_none()
        if not global_tenant:
            global_tenant = Tenant(id=0, name="Global")
            session.add(global_tenant)
            await session.flush()
            logger.info("Created global shared tenant (tenant_id=0)")

        # 2. 检查是否已有其他租户
        result = await session.execute(select(Tenant).where(Tenant.id != 0))
        tenant = result.scalar_one_or_none()

        if not tenant:
            # 创建默认租户
            tenant = Tenant(name="Default Tenant")
            session.add(tenant)
            await session.flush()
            logger.info("Created default tenant")

        # 检查是否已有用户
        result = await session.execute(select(User).where(User.username == "admin"))
        existing_user = result.scalar_one_or_none()

        if not existing_user:
            # 创建用户列表
            users_data = [
                {"username": "admin", "password": "admin123", "role": "admin", "desc": "系统管理员"},
                {"username": "manager_zhang", "password": "123456", "role": "supervisor", "desc": "部门主管-张经理"},
                {"username": "manager_li", "password": "123456", "role": "supervisor", "desc": "部门主管-李经理"},
                {"username": "employee_wang", "password": "123456", "role": "user", "desc": "普通员工-王五"},
                {"username": "employee_zhao", "password": "123456", "role": "user", "desc": "普通员工-赵六"},
                {"username": "employee_sun", "password": "123456", "role": "user", "desc": "普通员工-孙七"},
            ]

            for user_data in users_data:
                user = User(
                    tenant_id=tenant.id,
                    username=user_data["username"],
                    password_hash=get_password_hash(user_data["password"]),
                    role=user_data["role"]
                )
                session.add(user)
                logger.info(f"Created {user_data['desc']}: {user_data['username']} / {user_data['password']}")

        await session.commit()
        logger.info("Initialization complete!")


async def main():
    """
    主函数
    1. 创建所有数据库表
    2. 初始化默认数据
    """
    logger.info("开始初始化数据库...")

    # 创建表结构
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created")

    # 初始化默认数据
    await init_default_data()


if __name__ == "__main__":
    asyncio.run(main())