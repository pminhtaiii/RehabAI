import * as React from 'react';
import { useState } from 'react';
import { useNavigate } from "react-router-dom";
import { api } from '../api';

export default function Login() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [errorMessage, setErrorMessage] = useState('');
    const [successMessage, setSuccessMessage] = useState('');
    let navigate = useNavigate();

    const handleLogin = async (e) => {
        if(e) e.preventDefault();
        try {
            const response = await api.post('/api/login/', { username, password });
            console.log(response);
            navigate('/exercises')
        } catch (error) {
            console.error('Failed to login:', error);
            setErrorMessage(error.response?.data?.detail || "Login failed");
        }
    };

    const handleSignup = async (e) => {
        if(e) e.preventDefault();
        try {
            const response = await api.post('/api/signup/', { username, password });
            console.log(response);
            setSuccessMessage("Signup successful! You can now log in.");
            setErrorMessage('');
        } catch (error) {
            console.error('Failed to signup:', error);
            setErrorMessage(error.response?.data?.detail || "Signup failed");
        }
    };

    return (
        <div className="bg-surface text-on-surface min-h-screen flex flex-col font-['Inter']">
            {/* TopAppBar Component */}
            <nav className="w-full top-0 z-50 bg-[#fcf9f2]/80 backdrop-blur-[20px] flex justify-between items-center px-6 py-4 max-w-7xl mx-auto">
                <div className="flex items-center gap-2">
                    <span className="material-symbols-outlined text-[#153128]" data-icon="spa">spa</span>
                    <span className="text-2xl font-bold text-[#153128] tracking-tighter">RehabAI</span>
                </div>
                <div className="flex items-center gap-6">
                    <a className="text-[#424845] text-sm font-medium hover:opacity-80 transition-opacity" href="#">Support</a>
                    <a className="text-[#424845] text-sm font-medium hover:opacity-80 transition-opacity" href="#">Privacy</a>
                </div>
            </nav>

            {/* Main Content Canvas */}
            <main className="flex-grow flex flex-col items-center px-6 pt-12 pb-24 max-w-lg mx-auto w-full">
                {/* Hero Section: The Statement */}
                <header className="w-full mb-12 text-center md:text-left">
                    <h1 className="text-[3.5rem] leading-[1] font-extrabold text-primary tracking-tight mb-4" style={{letterSpacing: "-0.02em"}}>
                        Recover better, from anywhere
                    </h1>
                    <p className="text-[1.125rem] text-on-surface-variant leading-relaxed font-normal">
                        Your sanctuary for evidence-based rehabilitation. Let's continue your journey to full strength.
                    </p>
                </header>

                {/* Login Card */}
                <section className="w-full bg-surface-container-lowest ambient-shadow rounded-xl p-8 ghost-border relative overflow-hidden">
                    {/* Accent strip */}
                    <div className="absolute left-0 top-0 bottom-0 w-1 bg-secondary-fixed"></div>
                    <form className="space-y-8" onSubmit={handleLogin}>
                        {/* Username Field */}
                        <div className="space-y-1">
                            <label className="text-[0.75rem] uppercase tracking-[0.05em] font-medium text-on-surface-variant">Username</label>
                            <input 
                                className="w-full bg-surface-container-high border-0 border-b border-outline/20 px-4 py-3 text-on-surface focus:ring-0 focus:border-secondary focus:bg-surface-container-lowest transition-all" 
                                placeholder="your username" 
                                type="text"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                            />
                        </div>
                        {/* Password Field */}
                        <div className="space-y-1">
                            <label className="text-[0.75rem] uppercase tracking-[0.05em] font-medium text-on-surface-variant">Password</label>
                            <input 
                                className="w-full bg-surface-container-high border-0 border-b border-outline/20 px-4 py-3 text-on-surface focus:ring-0 focus:border-secondary focus:bg-surface-container-lowest transition-all" 
                                placeholder="••••••••" 
                                type="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                            />
                        </div>

                        {errorMessage && (
                            <div className="text-error text-sm font-medium mt-2">{errorMessage}</div>
                        )}
                        {successMessage && (
                            <div className="text-secondary text-sm font-medium mt-2">{successMessage}</div>
                        )}

                        {/* Action Group */}
                        <div className="space-y-6 pt-4">
                            <button className="w-full signature-texture text-on-primary py-4 rounded-xl font-bold text-lg hover:opacity-90 transition-all active:scale-95 duration-200" type="submit">
                                Log in
                            </button>
                            <div className="text-center">
                                <span className="text-on-surface-variant text-sm">New to RehabAI? </span>
                                <button type="button" onClick={handleSignup} className="text-secondary font-semibold hover:underline decoration-secondary/30">
                                    Create account
                                </button>
                            </div>
                        </div>
                    </form>
                </section>

                {/* Decorative Progress Bloom */}
                <div className="mt-16 w-32 h-32 rounded-full blur-[64px] opacity-20 bg-gradient-to-tr from-secondary to-tertiary-fixed pointer-events-none absolute -z-10"></div>
            </main>

            {/* Footer Component */}
            <footer className="w-full py-12 bg-[#f6f3ec] dark:bg-stone-950 mt-auto">
                <div className="flex flex-col md:flex-row justify-between items-center px-8 gap-4 max-w-7xl mx-auto">
                    <div className="text-[0.75rem] uppercase tracking-[0.05em] font-['Inter'] text-[#424845]">
                        © 2026 RehabAI Sanctuary. All rights reserved.
                    </div>
                    <div className="flex gap-8">
                        <a className="text-[0.75rem] uppercase tracking-[0.05em] font-['Inter'] text-[#424845] hover:text-[#153128] dark:hover:text-[#fcf9f2] transition-colors" href="#">Support</a>
                        <a className="text-[0.75rem] uppercase tracking-[0.05em] font-['Inter'] text-[#424845] hover:text-[#153128] dark:hover:text-[#fcf9f2] transition-colors" href="#">Privacy Policy</a>
                        <a className="text-[0.75rem] uppercase tracking-[0.05em] font-['Inter'] text-[#424845] hover:text-[#153128] dark:hover:text-[#fcf9f2] transition-colors" href="#">Terms of Service</a>
                    </div>
                </div>
            </footer>
        </div>
    )
}