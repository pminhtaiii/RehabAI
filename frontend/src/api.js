import axios from 'axios';

export const api = axios.create({
    baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
    // Allow sending cookies with requests (needed for authentication)
    withCredentials: true,
});