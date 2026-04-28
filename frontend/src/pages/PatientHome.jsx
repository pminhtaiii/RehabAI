import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api';

export default function PatientHome() {
    const [assignedExercises, setAssignedExercises] = useState([]);

    // Fetch the exercises assigned to the patient
    useEffect(() => {
        api.get('/api/exercises/')
            .then(res => setAssignedExercises(res.data))
            .catch(err => console.error(err));
    }, [])

    return (
        <div className="bg-background text-on-surface min-h-screen font-['Inter']">
            {/* TopAppBar */}
            <header className="fixed top-0 w-full z-50 bg-[#fcf9f2]/80 backdrop-blur-[20px]">
                <div className="flex justify-between items-center px-6 py-4 max-w-7xl mx-auto">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-[#2c473e] flex items-center justify-center overflow-hidden">
                            <span className="material-symbols-outlined text-white">person</span>
                        </div>
                        <span className="text-2xl font-black tracking-tighter text-[#153128]">RehabAI</span>
                    </div>
                    <div className="flex items-center gap-6">
                        <nav className="hidden md:flex items-center gap-8">
                            <a className="text-[#153128] font-bold text-sm tracking-wide" href="#">Exercises</a>
                            <a className="text-[#424845] font-medium text-sm tracking-wide hover:bg-[#f6f3ec] transition-colors duration-300 px-3 py-1 rounded-lg" href="#">Progress</a>
                            <a className="text-[#424845] font-medium text-sm tracking-wide hover:bg-[#f6f3ec] transition-colors duration-300 px-3 py-1 rounded-lg" href="#">Messages</a>
                        </nav>
                        <button className="w-10 h-10 flex items-center justify-center rounded-full hover:bg-[#f6f3ec] transition-colors duration-300">
                            <span className="material-symbols-outlined text-[#153128]">help</span>
                        </button>
                    </div>
                </div>
            </header>

            <main className="pt-24 pb-12 px-6 max-w-7xl mx-auto">
                {/* Hero Section */}
                <section className="mt-12 mb-16 grid grid-cols-1 lg:grid-cols-12 gap-8 items-end">
                    <div className="lg:col-span-8">
                        <h1 className="text-[3.5rem] leading-[1.1] font-extrabold text-primary tracking-tight mb-4">
                            Your assigned <br/>exercises
                        </h1>
                        <div className="flex items-center gap-4">
                            <div className="w-16 h-16 rounded-full flex items-center justify-center shadow-lg shadow-secondary/10" style={{background: "radial-gradient(circle at center, #a0f880 0%, #ccee9c 100%)"}}>
                                <span className="text-secondary font-bold text-lg">0%</span>
                            </div>
                            <p className="text-on-surface-variant text-lg font-medium">
                                Ready to start your session? <span className="text-secondary">Keep going!</span>
                            </p>
                        </div>
                    </div>
                </section>

                <div className="space-y-12">
                    {/* Featured Card: Recommended Next */}
                    {assignedExercises.length > 0 && (
                        <section>
                            <label className="text-[0.75rem] uppercase tracking-[0.05em] font-bold text-primary mb-6 block">Up Next</label>
                            <div className="bg-surface-container-low rounded-3xl overflow-hidden flex flex-col lg:flex-row shadow-sm">
                                <div className="lg:w-1/2 h-64 lg:h-auto overflow-hidden">
                                    <img alt="Featured Exercise" className="w-full h-full object-cover" src={assignedExercises[0].image_url} />
                                </div>
                                <div className="lg:w-1/2 p-8 lg:p-12 flex flex-col justify-center space-y-6">
                                    <div className="space-y-2">
                                        <div className="flex gap-2">
                                            <span className="bg-secondary-fixed text-on-secondary-fixed text-[0.65rem] font-bold px-2 py-1 rounded uppercase tracking-wider">Recommended</span>
                                            <span className="bg-surface-container-highest text-on-surface-variant text-[0.65rem] font-bold px-2 py-1 rounded uppercase tracking-wider">15 min</span>
                                        </div>
                                        <h2 className="text-3xl font-bold text-on-surface">{assignedExercises[0].name}</h2>
                                        <p className="text-on-surface-variant leading-relaxed max-w-md">Continue your recovery journey with this targeted exercise.</p>
                                    </div>
                                    <div className="flex flex-wrap gap-4 pt-4">
                                        <Link to={`/exercises/${assignedExercises[0].id}`} state={{exercise: assignedExercises[0]}} className="signature-texture text-white px-8 py-4 rounded-xl font-bold flex items-center justify-center gap-2 hover:opacity-90 transition-opacity">
                                            Start Exercise
                                            <span className="material-symbols-outlined text-sm">play_arrow</span>
                                        </Link>
                                    </div>
                                </div>
                            </div>
                        </section>
                    )}

                    {/* Grid of Exercises */}
                    <section>
                        <div className="flex justify-between items-center mb-8">
                            <label className="text-[0.75rem] uppercase tracking-[0.05em] font-bold text-primary">All Assigned Tasks</label>
                            <button className="text-secondary font-bold text-sm flex items-center gap-1">
                                Filter by Area
                                <span className="material-symbols-outlined text-sm">keyboard_arrow_down</span>
                            </button>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                            {assignedExercises.map((exercise, index) => (
                                <Link to={`/exercises/${exercise.id}`} state={{exercise}} key={index} className="group">
                                    <div className="bg-surface-container-low rounded-2xl p-4 transition-all duration-300 hover:bg-surface-container-high h-full flex flex-col">
                                        <div className="relative h-48 rounded-xl overflow-hidden mb-6 flex-shrink-0 bg-stone-100">
                                            <img className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" src={exercise.image_url} alt={exercise.name} />
                                            <div className="absolute top-3 left-3 flex gap-2">
                                                <span className="bg-white/90 backdrop-blur-md text-primary text-[0.6rem] font-bold px-2 py-1 rounded uppercase">Clinical</span>
                                            </div>
                                        </div>
                                        <div className="space-y-4 flex flex-col flex-grow">
                                            <div className="flex-grow">
                                                <h3 className="text-lg font-bold text-on-surface line-clamp-2 leading-tight h-10 mb-2">{exercise.name}</h3>
                                                <div className="flex items-center gap-4 text-xs font-medium text-on-surface-variant">
                                                    <span className="flex items-center gap-1"><span className="material-symbols-outlined text-[14px]">schedule</span> 10 min</span>
                                                </div>
                                            </div>
                                            <div className="flex items-center justify-between pt-4 border-t border-outline-variant/20 mt-auto">
                                                <span className="text-xs font-bold text-secondary uppercase tracking-widest">Pending</span>
                                                <button className="text-primary group-hover:text-secondary transition-colors">
                                                    <span className="material-symbols-outlined">arrow_forward</span>
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                </Link>
                            ))}
                        </div>
                    </section>
                </div>
            </main>

            {/* Footer */}
            <footer className="w-full rounded-t-[2rem] mt-12 bg-[#f6f3ec] dark:bg-stone-900 border-t border-stone-200 dark:border-stone-800">
                <div className="flex flex-col md:flex-row justify-between items-center px-10 py-12 gap-6 max-w-7xl mx-auto">
                    <div className="flex flex-col gap-2 text-center md:text-left">
                        <p className="text-[0.75rem] uppercase tracking-[0.05em] font-medium font-['Inter'] text-[#424845] dark:text-stone-400">© 2026 RehabAI Sanctuary. All rights reserved.</p>
                        <p className="text-[0.65rem] text-outline italic">Designed for clinical excellence and human recovery.</p>
                    </div>
                </div>
            </footer>
        </div>
    );
}