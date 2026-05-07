import { createRouter, createWebHistory } from 'vue-router'
import { useUserStore } from './stores/user'

const routes = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('./views/Login.vue')
  },
  {
    path: '/register',
    name: 'Register',
    component: () => import('./views/Register.vue')
  },
  {
    path: '/',
    redirect: '/chat'
  },
  {
    path: '/chat',
    name: 'Chat',
    component: () => import('./views/Chat.vue'),
    meta: { requiresAuth: true }
  },
  {
    path: '/documents',
    name: 'Documents',
    component: () => import('./views/Document.vue'),
    meta: { requiresAuth: true }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach((to, from, next) => {
  const userStore = useUserStore()
  const token = localStorage.getItem('token')

  if (token && !userStore.user) {
    userStore.setToken(token)
  }

  if (to.meta.requiresAuth && !token) {
    next('/login')
  } else if (to.path === '/login' && token) {
    next('/chat')
  } else {
    next()
  }
})

export default router