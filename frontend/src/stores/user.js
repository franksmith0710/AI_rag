import { defineStore } from 'pinia'
import { ref } from 'vue'
import api from '../api'

export const useUserStore = defineStore('user', () => {
  const token = ref(localStorage.getItem('token') || '')
  const user = ref(null)

  const setToken = (newToken) => {
    token.value = newToken
    localStorage.setItem('token', newToken)
    api.defaults.headers.common['Authorization'] = `Bearer ${newToken}`
  }

  const setUser = (newUser) => {
    user.value = newUser
  }

  const login = async (username, password) => {
    const response = await api.post('/api/auth/login', { username, password })
    setToken(response.data.access_token)
    setUser(response.data.user)
    return response.data
  }

  const register = async (username, password) => {
    const response = await api.post('/api/auth/register', {
      username,
      password,
      tenant_id: 0,
      role: 'user'
    })
    setToken(response.data.access_token)
    setUser(response.data.user)
    return response.data
  }

  const logout = () => {
    token.value = ''
    user.value = null
    localStorage.removeItem('token')
  }

  const fetchCurrentUser = async () => {
    if (!token.value) return null
    api.defaults.headers.common['Authorization'] = `Bearer ${token.value}`
    try {
      const response = await api.get('/api/auth/me')
      setUser(response.data)
      return response.data
    } catch {
      logout()
      return null
    }
  }

  return { token, user, setToken, setUser, login, register, logout, fetchCurrentUser }
})