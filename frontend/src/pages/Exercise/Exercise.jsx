import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import * as poseDetection from '@tensorflow-models/pose-detection';
import { MiniDataFrame } from './utils';
import { RendererCanvas2d } from './renderer_canvas2d';
import { csvToJSON } from './utils';
import { api } from '../../api';

// Keypoint dictionary
const KEYPOINT_DICT = {
  'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
  'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
  'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
  'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
};

export default function Exercise() {
    const location = useLocation();
    const exerciseInfo = location.state?.exercise;

    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const referenceVideoRef = useRef(null);
    const reqAFRef = useRef(null);

    // AI Refs to avoid re-rendering
    const movenetRef = useRef(null);
    const rendererRef = useRef(null);
    const referenceDFRef = useRef(null);
    
    // Performance: Use array instead of pandas-js dataframe for 60FPS pushes
    const framesRef = useRef([]); 
    
    // State
    const [isSaving, setIsSaving] = useState(false);
    const [isExerciseFinished, setIsExerciseFinished] = useState(false);
    const [countdown, setCountdown] = useState(null);
    const [elapsedTime, setElapsedTime] = useState(0);
    const [feedbackMessages, setFeedbackMessages] = useState([]);
    const [currentFeedbackMessages, setCurrentFeedbackMessages] = useState([]);
    const [clinicalScore, setClinicalScore] = useState(null);
    const [liveScore, setLiveScore] = useState(null);
    const [feedbackLatency, setFeedbackLatency] = useState(null);
    const [motionDetected, setMotionDetected] = useState(null);
    const [motionEnergy, setMotionEnergy] = useState(null);

    const [videoLoaded, setVideoLoaded] = useState(false);
    const isSavingRef = useRef(false);
    const wsRef = useRef(null);
    const progressiveScoreIntervalRef = useRef(null);

    useEffect(() => {
        isSavingRef.current = isSaving;
    }, [isSaving]);

    // Initialization
    useEffect(() => {
        if (!exerciseInfo) return;

        let isMounted = true;
        const initCameraAndAI = async () => {
             // 1. Setup Camera
             try {
                 const stream = await navigator.mediaDevices.getUserMedia({
                     audio: false,
                     video: { facingMode: 'user', width: 640, height: 480, frameRate: { ideal: 30 } }
                 });
                 if (videoRef.current) {
                     videoRef.current.srcObject = stream;
                     videoRef.current.onloadedmetadata = () => {
                         videoRef.current.play();
                         setVideoLoaded(true);
                     };
                 }
             } catch (e) {
                 console.error("Camera error", e);
             }

             // 2. Setup TFJS & MoveNet
             await tf.setBackend('webgl');
             await tf.ready();
             // Performance Fix: Use LIGHTNING model for real-time smoothness
             const detectorConfig = { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING };
             movenetRef.current = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
             if (canvasRef.current) {
                 rendererRef.current = new RendererCanvas2d(canvasRef.current);
             }

             // 3. Load Reference Data
             try {
                const response = await fetch(exerciseInfo.csv);
                const csvText = await response.text();
                referenceDFRef.current = new MiniDataFrame(csvToJSON(csvText));
             } catch (e) { console.error("CSV loading error", e); }

             if (isMounted) detectPose();
        };

        initCameraAndAI();

        return () => {
            isMounted = false;
            if (reqAFRef.current) cancelAnimationFrame(reqAFRef.current);
            if (videoRef.current && videoRef.current.srcObject) {
                videoRef.current.srcObject.getTracks().forEach(t => t.stop());
            }
        };
    }, [exerciseInfo]);

    // 4. The Pose Detection Loop
    const detectPose = () => {
        const videoWidth = 640, videoHeight = 480;

        const findPose = async () => {
            if (!videoRef.current || !movenetRef.current || !canvasRef.current || videoRef.current.readyState !== 4) {
                reqAFRef.current = requestAnimationFrame(findPose);
                return;
            }

            const video = videoRef.current;
            const canvas = canvasRef.current;
            if(canvas.width !== videoWidth) {
                canvas.width = videoWidth;
                canvas.height = videoHeight;
            }
            const ctx = canvas.getContext('2d');
            
            const poses = await movenetRef.current.estimatePoses(video, { flipHorizontal: false, flipVertical: false });
            
            if (isSavingRef.current && poses[0]) {
                const normalizedKeypoints = poseDetection.calculators.keypointsToNormalizedKeypoints(poses[0].keypoints, video);
                
                const frameData = {};
                for (const [jointName, jointIndex] of Object.entries(KEYPOINT_DICT)) {
                    const keypoint = normalizedKeypoints[jointIndex];
                    if (keypoint) {
                        frameData[jointName + '_x'] = keypoint.x;
                        frameData[jointName + '_y'] = keypoint.y;
                        frameData[jointName + '_confidence'] = keypoint.score;
                    }
                }
                framesRef.current.push(frameData);

                // Every 30 frames, do DTW comparison
                if (framesRef.current.length % 30 === 0 && referenceDFRef.current) {
                   compareJointsWithReference(framesRef.current);
                }
            }

            ctx.clearRect(0, 0, videoWidth, videoHeight);
            ctx.save();
            ctx.scale(-1, 1);
            ctx.translate(-videoWidth, 0);
            ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
            if (rendererRef.current && poses.length > 0) rendererRef.current.drawResults(poses);
            ctx.restore();
            
            reqAFRef.current = requestAnimationFrame(findPose);
        };
        findPose();
    };

    // 5. WebSocket-based feedback (replaces REST polling for lower latency)
    const connectWebSocket = (exerciseId) => {
        const baseUrl = (import.meta.env.VITE_API_URL || 'http://localhost:8000')
            .replace(/^http/, 'ws');
        const ws = new WebSocket(`${baseUrl}/ws/session/${exerciseId}`);

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'feedback') {
                handleFeedbackResponse(data);
                if (data.latency_ms) setFeedbackLatency(data.latency_ms);
            } else if (data.type === 'clinical_score') {
                setLiveScore(data.score);
                if (data.motion_detected !== undefined) setMotionDetected(data.motion_detected);
                if (data.motion_energy !== undefined) setMotionEnergy(data.motion_energy);
            } else if (data.type === 'error') {
                console.error('WebSocket error:', data.message);
            }
        };

        ws.onerror = (e) => console.error('WebSocket connection error:', e);
        ws.onclose = () => console.log('WebSocket closed');
        wsRef.current = ws;
    };

    const disconnectWebSocket = () => {
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
    };

    const handleFeedbackResponse = (data) => {
        let newFeedback = [];
        if (data.feedback_details) {
            for (const item of data.feedback_details) {
                if (item.status === 'needs_correction') {
                    newFeedback.push({
                        message: item.message,
                        joint: item.joint_group,
                        severity: item.severity,
                    });
                }
            }
        }

        if (data.primary_instruction && newFeedback.length > 0) {
            newFeedback.unshift({ message: data.primary_instruction, severity: 1 });
        }

        if (newFeedback.length > 0) {
            setCurrentFeedbackMessages(newFeedback);
            setFeedbackMessages(prev => [...prev, ...newFeedback]);
        } else {
            setCurrentFeedbackMessages([]);
        }
    };

    const buildKeypointArray = (frameData) => {
        const KEYPOINT_ORDER = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
        ];
        const values = [];
        for (const name of KEYPOINT_ORDER) {
            values.push(frameData[name + '_y'] ?? 0);
            values.push(frameData[name + '_x'] ?? 0);
            values.push(frameData[name + '_confidence'] ?? 0);
        }
        return values;
    };

    const compareJointsWithReference = async (framesArray) => {
        if (!referenceDFRef.current) return;
        const currentFrameIndex = framesArray.length - 1;

        const currentFrame = framesArray[currentFrameIndex];
        const currentJointValues = buildKeypointArray(currentFrame);

        const refData = referenceDFRef.current.data;
        const refIndex = Math.min(currentFrameIndex, refData.length - 1);
        const refFrame = refData[refIndex];
        const referenceJointValues = buildKeypointArray(refFrame);

        // Use WebSocket if available, fallback to REST
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({
                type: 'feedback',
                referenceJointValues,
                currentJointValues,
            }));
        } else {
            // REST fallback
            try {
                const response = await api.post(`/api/feedback/${exerciseInfo.exercise_id}/`, {
                    referenceJointValues,
                    currentJointValues,
                    includeDtw: false,
                });
                handleFeedbackResponse(response.data);
            } catch (e) {
                console.error('Feedback API error:', e);
            }
        }
    };

    const fetchClinicalScore = async () => {
        try {
            const finalDF = new MiniDataFrame(framesRef.current);
            const response = await api.post(`/api/clinical_score/${exerciseInfo?.exercise_id}`, {
                csvString: finalDF.to_csv(),
            });
            if (response.data) {
                setClinicalScore(response.data.clinical_score[0][0]);
                if (response.data.motion_detected !== undefined) setMotionDetected(response.data.motion_detected);
                if (response.data.motion_energy !== undefined) setMotionEnergy(response.data.motion_energy);
            }
        } catch (e) { console.error(e); }
    };

    // 6. Triggers 
    const timerIntervalRef = useRef(null);
    const exerciseTimerRef = useRef(null);
    
    const startExerciseCountDown = () => {
        setCountdown(10);
        const inv = setInterval(() => {
            setCountdown(c => {
                if (c <= 1) {
                    clearInterval(inv);
                    startExerciseSession();
                    return null;
                }
                return c - 1;
            });
        }, 1000);
    };

    const startExerciseSession = () => {
        setIsSaving(true);
        setIsExerciseFinished(false);
        framesRef.current = [];
        setElapsedTime(0);
        setCurrentFeedbackMessages([]);
        setFeedbackMessages([]);
        setClinicalScore(null);
        setLiveScore(null);
        setFeedbackLatency(null);
        setMotionDetected(null);
        setMotionEnergy(null);
        
        // Connect WebSocket for real-time feedback
        connectWebSocket(exerciseInfo.exercise_id);
        
        if (referenceVideoRef.current) {
            referenceVideoRef.current.currentTime = 0;
            referenceVideoRef.current.play();
        }

        const startTimestamp = Date.now();
        // Performance Fix: decouple stopwatch from 60FPS react re-renders
        timerIntervalRef.current = setInterval(() => {
            setElapsedTime((Date.now() - startTimestamp) / 1000);
        }, 500); 

        // Progressive clinical score: request score every 5 seconds
        progressiveScoreIntervalRef.current = setInterval(() => {
            if (framesRef.current.length > 30 && wsRef.current?.readyState === WebSocket.OPEN) {
                const partialDF = new MiniDataFrame(framesRef.current);
                wsRef.current.send(JSON.stringify({
                    type: 'clinical_score',
                    csvString: partialDF.to_csv(),
                }));
            }
        }, 5000);

        exerciseTimerRef.current = setTimeout(() => {
            stopExerciseSession();
        }, 30000); // Wait 30s
    };

    const stopExerciseSession = () => {
        setIsSaving(false);
        if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
        if (exerciseTimerRef.current) clearTimeout(exerciseTimerRef.current);
        if (progressiveScoreIntervalRef.current) clearInterval(progressiveScoreIntervalRef.current);
        if (referenceVideoRef.current) referenceVideoRef.current.pause();
        
        disconnectWebSocket();
        
        setIsExerciseFinished(true);
        fetchClinicalScore();
    };

    if (!exerciseInfo) return <div>No Exercise Data Provided. Please Navigate back.</div>;

    // Formatting 
    const isReady = videoLoaded; // camera is on
    const timeLeft = 30 - Math.floor(elapsedTime);
    let displayTime = isSaving ? `00:${timeLeft < 10 ? '0'+timeLeft : timeLeft}` : "00:00";

    return (
        <div className="bg-background text-on-background min-h-screen pb-20 font-['Inter']">
            {/* TopAppBar */}
            <header className="fixed top-0 w-full z-50 bg-[#fcf9f2]/80 backdrop-blur-[20px] shadow-sm">
                <div className="flex justify-between items-center px-6 py-4 w-full max-w-[1600px] mx-auto">
                    <Link to="/exercises" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
                        <span className="material-symbols-outlined text-[#2c473e]">arrow_back</span>
                        <h1 className="text-[#2c473e] font-bold tracking-tighter text-xl">RehabAI</h1>
                    </Link>
                    <div className="flex items-center gap-4">
                        <button className="p-2 hover:bg-[#ebe8e1] transition-colors duration-200 rounded-full active:scale-95">
                            <span className="material-symbols-outlined text-[#424845]">help_outline</span>
                        </button>
                    </div>
                </div>
            </header>

            <main className="pt-24 px-6 w-full max-w-[1600px] mx-auto flex flex-col">
                {/* Compact Hero Section */}
                <section className="mb-6">
                    <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
                        <div className="space-y-2">
                            <div className="flex items-center gap-3 mb-2">
                                <span className="px-3 py-1 bg-secondary-container text-on-secondary-container text-[12px] uppercase tracking-[0.05em] font-semibold rounded-full">Active Session</span>
                                <span className="text-on-surface-variant font-medium text-[14px]">Rehabilitation</span>
                            </div>
                            <h2 className="text-[40px] md:text-[60px] font-extrabold tracking-tighter leading-tight text-primary">{exerciseInfo.name || "Targeted Exercise"}</h2>
                            <p className="text-on-surface-variant text-lg font-medium flex items-center gap-2">
                                <span className="material-symbols-outlined text-[20px]">schedule</span>
                                30s rep
                            </p>
                        </div>
                    </div>
                </section>

                {/* Main Content Layout */}
                <div className="flex flex-col gap-6 w-full max-w-[1600px] mx-auto pb-12">
                    {/* Top Row: Video Feeds */}
                    <div className="w-full">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 h-[60vh] min-h-[400px]">
                            {/* Reference Exercise Feed */}
                            <div className="bg-surface-container-low rounded-xl overflow-hidden relative group h-full">
                                <video 
                                    ref={referenceVideoRef} 
                                    src={exerciseInfo.video_url} 
                                    loop 
                                    playsInline 
                                    muted 
                                    className="w-full h-full object-cover grayscale-[20%]" 
                                />
                                <div className="absolute inset-0 bg-gradient-to-t from-primary/40 to-transparent pointer-events-none"></div>
                                <div className="absolute bottom-6 left-6 flex items-center gap-3">
                                    <span className="text-white font-semibold">Reference Technique</span>
                                </div>
                                <div className="absolute top-6 right-6 px-3 py-1 bg-white/90 backdrop-blur-sm rounded-full text-xs font-bold text-primary">PRO VIEW</div>
                            </div>

                            {/* Live User Feed Canvas */}
                            <div className="bg-black rounded-xl relative flex items-center justify-center overflow-hidden border-2 border-primary/20 bg-stone-900 h-full">
                                <video ref={videoRef} playsInline hidden />
                                <canvas ref={canvasRef} className="w-full h-full object-contain"></canvas>
                                {isSaving && (
                                    <div className="absolute top-4 left-4 flex gap-2">
                                        <div className="px-3 py-1 bg-red-500 text-white rounded-full text-xs font-bold uppercase flex items-center gap-2 animate-pulse">
                                            <div className="w-2 h-2 rounded-full bg-white"></div>
                                            Recording
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>

                        <div className="flex gap-3 justify-center mt-6">
                            {!isSaving && !countdown && (
                                <button onClick={startExerciseCountDown} disabled={!isReady} className={`px-8 py-4 text-white font-bold rounded-lg shadow-sm transition-transform active:scale-95 flex items-center gap-2 ${isReady ? 'bg-[#6ABE4E] hover:bg-[#5aa842]' : 'bg-gray-400'}`}>
                                    <span className="material-symbols-outlined" style={{fontVariationSettings: "'FILL' 1"}}>play_arrow</span>
                                    {isReady ? "Start Exercise" : "Starting Camera..."}
                                </button>
                            )}
                            {countdown && (
                                <div className="px-8 py-4 bg-primary text-white font-bold rounded-lg shadow-sm flex items-center gap-2">
                                    Starting in {countdown}...
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Bottom Row: Instructions & Feedback Sidebar */}
                    <div className="w-full grid grid-cols-1 md:grid-cols-2 gap-8 mt-2">
                        {/* Stats Column */}
                        <div className="flex flex-col gap-6">
                            {/* Progress & Stats Card */}
                        <div className="bg-surface-container-lowest p-6 rounded-lg shadow-[0_20px_40px_rgba(21,49,40,0.04)]">
                            <div className="flex justify-between items-center mb-6">
                                <div>
                                    <p className="text-[10px] uppercase tracking-[0.1em] font-bold text-on-surface-variant mb-1">Clinical Score</p>
                                    <p className="text-3xl font-extrabold text-primary">
                                        {clinicalScore !== null 
                                            ? Number(clinicalScore).toFixed(0) 
                                            : liveScore !== null 
                                                ? <span className="animate-pulse">{Number(liveScore).toFixed(0)}</span>
                                                : "--"}
                                        <span className="text-sm font-medium text-on-surface-variant ml-1">/100</span>
                                    </p>
                                    {feedbackLatency !== null && isSaving && (
                                        <p className="text-[9px] text-on-surface-variant mt-1">
                                            {feedbackLatency.toFixed(0)}ms latency
                                        </p>
                                    )}
                                </div>
                                <div className="text-right">
                                    <p className="text-[10px] uppercase tracking-[0.1em] font-bold text-on-surface-variant mb-1">Time Remaining</p>
                                    <p className="text-3xl font-mono font-bold text-secondary">{displayTime}</p>
                                </div>
                            </div>
                            {/* Progress bar */}
                            <div className="h-2 w-full bg-surface-container rounded-full overflow-hidden">
                                <div className="h-full bg-[#6ABE4E] rounded-full transition-all duration-500" style={{width: `${Math.min(100, (elapsedTime / 30) * 100)}%`}}></div>
                            </div>

                            {/* Live Motion Energy Indicator */}
                            {isSaving && motionEnergy !== null && (
                                <div className="flex items-center gap-3 mt-3 px-1">
                                    <span className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant whitespace-nowrap">Motion Level</span>
                                    <div className="h-1.5 flex-1 bg-surface-container rounded-full overflow-hidden">
                                        <div className={`h-full rounded-full transition-all duration-700 ${
                                            motionEnergy > 0.5 ? 'bg-[#6ABE4E]' : motionEnergy > 0.2 ? 'bg-amber-400' : 'bg-red-400'
                                        }`} style={{width: `${Math.min(100, motionEnergy * 100)}%`}}></div>
                                    </div>
                                    <span className={`text-[10px] font-bold ${
                                        motionDetected ? 'text-[#6ABE4E]' : 'text-amber-500'
                                    }`}>{motionDetected ? 'Active' : 'Low'}</span>
                                </div>
                            )}
                        </div>

                        {/* Secondary Actions */}
                        {isSaving && (
                            <div className="grid grid-cols-2 gap-3 pt-4">
                                <button onClick={() => {}} className="py-3 px-2 bg-surface-container-high hover:bg-surface-variant rounded-lg text-xs font-bold text-on-surface-variant transition-colors flex flex-col items-center gap-1">
                                    <span className="material-symbols-outlined text-lg">pause</span> Pause
                                </button>
                                <button onClick={stopExerciseSession} className="py-3 px-2 bg-error-container/30 hover:bg-error-container text-error rounded-lg text-xs font-bold transition-colors flex flex-col items-center gap-1">
                                    <span className="material-symbols-outlined text-lg">stop</span> Stop
                                </button>
                            </div>
                        )}
                    </div>

                    {/* Feedback Column */}
                    <div className="flex flex-col gap-6">
                        {/* Live Feedback Cards */}
                        <div className="flex flex-col gap-4">
                            <h3 className="text-sm font-bold uppercase tracking-widest text-on-surface-variant px-1">Real-time Guidance</h3>

                            {isSaving && currentFeedbackMessages.length === 0 && (
                                <div className="bg-[#B0D182]/20 border-l-4 border-[#6ABE4E] p-4 rounded-lg flex items-start gap-4">
                                    <span className="material-symbols-outlined text-[#6ABE4E]" style={{fontVariationSettings: "'FILL' 1"}}>check_circle</span>
                                    <div>
                                        <p className="font-bold text-[#153128]">Perfect Alignment</p>
                                        <p className="text-sm text-on-surface-variant mt-1">Movement matches reference frame optimally.</p>
                                    </div>
                                </div>
                            )}

                            {currentFeedbackMessages.map((msg, i) => (
                                <div key={i} className="bg-surface-variant/50 p-4 rounded-lg flex items-start gap-4 border-l-4 border-amber-500">
                                    <span className="material-symbols-outlined text-amber-500">info</span>
                                    <div>
                                        <p className="font-bold text-primary">Adjustment Needed</p>
                                        <p className="text-sm text-on-surface-variant mt-1">{msg.message}</p>
                                    </div>
                                </div>
                            ))}

                            {!isSaving && !clinicalScore && (
                                <div className="bg-surface-container-low p-4 rounded-lg text-sm text-on-surface-variant">
                                    Start the exercise to receive live AI posture feedback.
                                </div>
                            )}
                            {clinicalScore !== null && (
                                <div className="p-4 rounded-lg flex items-start gap-4 border-l-4 bg-secondary-container/50 border-secondary">
                                    <span className="material-symbols-outlined text-secondary" style={{fontVariationSettings: "'FILL' 1"}}>
                                        verified
                                    </span>
                                    <div>
                                        <p className="font-bold text-primary">Session Finished</p>
                                        <p className="text-sm text-on-surface-variant mt-1">Tuyệt vời! Điểm lâm sàng đã được ghi nhận.</p>
                                        {motionEnergy !== null && (
                                            <div className="mt-2 flex items-center gap-2">
                                                <span className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">Motion</span>
                                                <div className="h-1.5 w-20 bg-surface-container rounded-full overflow-hidden">
                                                    <div className={`h-full rounded-full transition-all ${
                                                        motionEnergy > 0.5 ? 'bg-[#6ABE4E]' : motionEnergy > 0.2 ? 'bg-amber-400' : 'bg-red-400'
                                                    }`} style={{width: `${Math.min(100, motionEnergy * 100)}%`}}></div>
                                                </div>
                                                <span className="text-[10px] text-on-surface-variant">{(motionEnergy * 100).toFixed(0)}%</span>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
                </div>
            </main>
        </div>
    );
}