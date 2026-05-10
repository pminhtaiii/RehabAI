import React, { useState, useEffect, useRef } from 'react';
import { useLocation, Link } from 'react-router-dom';
import { PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { MiniDataFrame } from './utils';
import { RendererCanvas2d } from './renderer_canvas2d';
import { api } from '../../api';

const BODY_LANDMARKS = {
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
};

/**
 * CSV column order for 17 KiMoRe keypoints (y, x, confidence × 17 = 51 columns).
 * Maps CSV column indices to the 12 BODY_LANDMARKS used by the renderer.
 * Each body landmark has 3 values: (y, x, confidence).
 */
const CSV_BODY_COL_OFFSETS = {
    // landmark_name: [y_col_offset, x_col_offset, confidence_col_offset]
    // Keypoint index in CSV: nose(0), left_eye(1), right_eye(2), left_ear(3), right_ear(4),
    //   left_shoulder(5), right_shoulder(6), left_elbow(7), right_elbow(8),
    //   left_wrist(9), right_wrist(10), left_hip(11), right_hip(12),
    //   left_knee(13), right_knee(14), left_ankle(15), right_ankle(16)
    'left_shoulder': [15, 16, 17],   // keypoint 5: cols 15-17
    'right_shoulder': [18, 19, 20],   // keypoint 6: cols 18-20
    'left_elbow': [21, 22, 23],   // keypoint 7: cols 21-23
    'right_elbow': [24, 25, 26],   // keypoint 8: cols 24-26
    'left_wrist': [27, 28, 29],   // keypoint 9: cols 27-29
    'right_wrist': [30, 31, 32],   // keypoint 10: cols 30-32
    'left_hip': [33, 34, 35],   // keypoint 11: cols 33-35
    'right_hip': [36, 37, 38],   // keypoint 12: cols 36-38
    'left_knee': [39, 40, 41],   // keypoint 13: cols 39-41
    'right_knee': [42, 43, 44],   // keypoint 14: cols 42-44
    'left_ankle': [45, 46, 47],   // keypoint 15: cols 45-47
    'right_ankle': [48, 49, 50],   // keypoint 16: cols 48-50
};

// Order matching BODY_LANDMARKS keys
const BODY_LANDMARK_NAMES = [
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
];

/**
 * Parse reference CSV text and extract 12 body landmarks from a representative frame.
 * CSV format: 17 keypoints × 3 (y, x, confidence) = 51 columns, ~520 rows.
 * @param {string} csvText - Raw CSV text
 * @param {number} frameIndex - Frame index to extract (default: middle frame)
 * @returns {Array<{x: number, y: number, confidence: number}>} 12 landmarks in BODY_LANDMARK_NAMES order
 */
function extractReferencePoseFromCSV(csvText, frameIndex = null) {
    const lines = csvText.trim().split('\n');
    if (lines.length < 2) return null;

    // Skip header row, pick middle frame if not specified
    const dataLines = lines.slice(1);
    const idx = frameIndex !== null ? frameIndex : Math.floor(dataLines.length / 2);
    const safeIdx = Math.min(idx, dataLines.length - 1);
    const row = dataLines[safeIdx].split(',').map(Number);

    if (row.length < 51) {
        console.warn('[extractReferencePose] CSV row has fewer than 51 columns:', row.length);
        return null;
    }

    const landmarks = [];
    for (const name of BODY_LANDMARK_NAMES) {
        const [yCol, xCol, confCol] = CSV_BODY_COL_OFFSETS[name];
        landmarks.push({
            x: row[xCol],
            y: row[yCol],
            confidence: row[confCol],
        });
    }
    return landmarks;
}

/**
 * Compute confidence level from quality_factor and motion_energy.
 * Returns {level: 'high'|'medium'|'low'|'none', color: string, label: string}
 */
function computeConfidenceDisplay(qualityFactor, motionEnergy, motionDetected) {
    if (!motionDetected || qualityFactor === null) {
        return { level: 'none', color: 'bg-gray-400', textColor: 'text-gray-500', label: 'No motion' };
    }
    // quality_factor is [0.1, 1.0], motion_energy is [0, 1]
    // Composite confidence: weighted combination
    const confidence = qualityFactor * 0.7 + (motionEnergy || 0) * 0.3;
    if (confidence >= 0.6) {
        return { level: 'high', color: 'bg-[#6ABE4E]', textColor: 'text-[#6ABE4E]', label: 'High' };
    } else if (confidence >= 0.3) {
        return { level: 'medium', color: 'bg-amber-400', textColor: 'text-amber-500', label: 'Medium' };
    } else {
        return { level: 'low', color: 'bg-red-400', textColor: 'text-red-500', label: 'Low' };
    }
}

export default function Exercise() {
    const location = useLocation();
    const exerciseInfo = location.state?.exercise;

    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const referenceVideoRef = useRef(null);
    const reqAFRef = useRef(null);

    const poseLandmarkerRef = useRef(null);
    const rendererRef = useRef(null);

    const framesRef = useRef([]);

    const [isSaving, setIsSaving] = useState(false);
    const [isExerciseFinished, setIsExerciseFinished] = useState(false);
    const [countdown, setCountdown] = useState(null);
    const [elapsedTime, setElapsedTime] = useState(0);
    const [clinicalScore, setClinicalScore] = useState(null);
    const [motionDetected, setMotionDetected] = useState(null);
    const [motionEnergy, setMotionEnergy] = useState(null);
    const [qualityFactor, setQualityFactor] = useState(null);
    const [isScoring, setIsScoring] = useState(false);
    const [rawScore, setRawScore] = useState(null);

    const [videoLoaded, setVideoLoaded] = useState(false);
    const isSavingRef = useRef(false);
    const lastTimestampRef = useRef(-1);

    useEffect(() => {
        isSavingRef.current = isSaving;
    }, [isSaving]);

    useEffect(() => {
        if (!exerciseInfo) return;

        let isMounted = true;
        const initCameraAndAI = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: false,
                    video: { facingMode: 'user', frameRate: { ideal: 30 } }
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

            const vision = await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
            );
            poseLandmarkerRef.current = await PoseLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
                    delegate: "GPU",
                },
                runningMode: "VIDEO",
                numPoses: 1,
            });

            if (canvasRef.current) {
                rendererRef.current = new RendererCanvas2d(canvasRef.current);
            }

            // Load reference CSV → extract representative frame → set silhouette
            try {
                const response = await fetch(exerciseInfo.csv);
                const csvText = await response.text();
                const refPose = extractReferencePoseFromCSV(csvText);
                if (refPose && rendererRef.current) {
                    rendererRef.current.setReferencePose(refPose);
                }
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
            if (poseLandmarkerRef.current) {
                poseLandmarkerRef.current.close();
            }
        };
    }, [exerciseInfo]);

    const detectPose = () => {
        const findPose = async () => {
            if (!videoRef.current || !poseLandmarkerRef.current || !canvasRef.current || videoRef.current.readyState !== 4) {
                reqAFRef.current = requestAnimationFrame(findPose);
                return;
            }

            const video = videoRef.current;
            const canvas = canvasRef.current;
            const videoWidth = video.videoWidth;
            const videoHeight = video.videoHeight;

            if (canvas.width !== videoWidth) {
                canvas.width = videoWidth;
                canvas.height = videoHeight;
                if (rendererRef.current) {
                    rendererRef.current.videoWidth = videoWidth;
                    rendererRef.current.videoHeight = videoHeight;
                    rendererRef.current.invalidateSilhouetteCache();
                }
            }
            const ctx = canvas.getContext('2d');

            const nowMs = performance.now();
            if (nowMs <= lastTimestampRef.current) {
                reqAFRef.current = requestAnimationFrame(findPose);
                return;
            }
            lastTimestampRef.current = nowMs;

            const result = poseLandmarkerRef.current.detectForVideo(video, nowMs);

            // Record frame data when session is active
            if (isSavingRef.current && result.landmarks && result.landmarks.length > 0) {
                const landmarks = result.landmarks[0];

                const TARGET_AR = 16 / 9;
                const actualAR = videoWidth / videoHeight;
                const xCorrectionFactor = actualAR / TARGET_AR;

                const frameData = {};
                for (const [jointName, landmarkIndex] of Object.entries(BODY_LANDMARKS)) {
                    const lm = landmarks[landmarkIndex];
                    if (lm) {
                        frameData[jointName + '_x'] = lm.x * xCorrectionFactor;
                        frameData[jointName + '_y'] = lm.y;
                        frameData[jointName + '_z'] = lm.z;
                        frameData[jointName + '_v'] = lm.visibility ?? 1.0;
                    }
                }
                framesRef.current.push(frameData);
            }

            // Rendering order: video → silhouette → user skeleton
            ctx.clearRect(0, 0, videoWidth, videoHeight);
            ctx.save();
            ctx.scale(-1, 1);
            ctx.translate(-videoWidth, 0);
            ctx.drawImage(video, 0, 0, videoWidth, videoHeight);

            // Draw silhouette overlay (translucent reference pose)
            if (rendererRef.current) {
                rendererRef.current.drawSilhouette(0.2);
            }

            // Draw user skeleton on top
            if (rendererRef.current && result.landmarks && result.landmarks.length > 0) {
                rendererRef.current.drawResults(result.landmarks);
            }
            ctx.restore();

            reqAFRef.current = requestAnimationFrame(findPose);
        };
        findPose();
    };

    const fetchClinicalScore = async () => {
        setIsScoring(true);
        try {
            const finalDF = new MiniDataFrame(framesRef.current);
            const response = await api.post(`/api/clinical_score/${exerciseInfo?.exercise_id}`, {
                csvString: finalDF.to_csv(),
            });
            if (response.data) {
                setClinicalScore(response.data.clinical_score[0][0]);
                setRawScore(response.data.raw_score || null);
                if (response.data.motion_detected !== undefined) setMotionDetected(response.data.motion_detected);
                if (response.data.motion_energy !== undefined) setMotionEnergy(response.data.motion_energy);
                if (response.data.quality_factor !== undefined) setQualityFactor(response.data.quality_factor);
            }
        } catch (e) { console.error(e); }
        setIsScoring(false);
    };

    const timerIntervalRef = useRef(null);
    const exerciseTimerRef = useRef(null);
    const countdownIntervalRef = useRef(null);

    const clearAllTimers = () => {
        if (countdownIntervalRef.current) {
            clearInterval(countdownIntervalRef.current);
            countdownIntervalRef.current = null;
        }
        if (timerIntervalRef.current) {
            clearInterval(timerIntervalRef.current);
            timerIntervalRef.current = null;
        }
        if (exerciseTimerRef.current) {
            clearTimeout(exerciseTimerRef.current);
            exerciseTimerRef.current = null;
        }
    };

    const startExerciseCountDown = () => {
        clearAllTimers();
        setCountdown(10);
        countdownIntervalRef.current = setInterval(() => {
            setCountdown(c => {
                if (c <= 1) {
                    clearInterval(countdownIntervalRef.current);
                    countdownIntervalRef.current = null;
                    startExerciseSession();
                    return null;
                }
                return c - 1;
            });
        }, 1000);
    };

    const startExerciseSession = () => {
        clearAllTimers();

        setIsSaving(true);
        setIsExerciseFinished(false);
        framesRef.current = [];
        setElapsedTime(0);
        setClinicalScore(null);
        setRawScore(null);
        setMotionDetected(null);
        setMotionEnergy(null);
        setQualityFactor(null);
        setIsScoring(false);

        if (referenceVideoRef.current) {
            referenceVideoRef.current.currentTime = 0;
            referenceVideoRef.current.play();
        }

        const startTimestamp = Date.now();
        timerIntervalRef.current = setInterval(() => {
            setElapsedTime((Date.now() - startTimestamp) / 1000);
        }, 500);

        exerciseTimerRef.current = setTimeout(() => {
            stopExerciseSession();
        }, 30000);
    };

    const stopExerciseSession = () => {
        setIsSaving(false);
        clearAllTimers();
        if (referenceVideoRef.current) referenceVideoRef.current.pause();

        setIsExerciseFinished(true);
        fetchClinicalScore();
    };

    if (!exerciseInfo) return <div>No Exercise Data Provided. Please Navigate back.</div>;

    const isReady = videoLoaded;
    const timeLeft = 30 - Math.floor(elapsedTime);
    let displayTime = isSaving ? `00:${timeLeft < 10 ? '0' + timeLeft : timeLeft}` : "00:00";

    const confidence = computeConfidenceDisplay(qualityFactor, motionEnergy, motionDetected);

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
                                {/* Silhouette guide label */}
                                <div className="absolute top-4 right-4 px-3 py-1 bg-white/80 backdrop-blur-sm rounded-full text-[10px] font-bold text-primary flex items-center gap-1">
                                    <span className="material-symbols-outlined text-[14px]">person</span>
                                    Follow the green outline
                                </div>
                            </div>
                        </div>

                        <div className="flex gap-3 justify-center mt-6">
                            {!isSaving && !countdown && (
                                <button onClick={startExerciseCountDown} disabled={!isReady} className={`px-8 py-4 text-white font-bold rounded-lg shadow-sm transition-transform active:scale-95 flex items-center gap-2 ${isReady ? 'bg-[#6ABE4E] hover:bg-[#5aa842]' : 'bg-gray-400'}`}>
                                    <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>play_arrow</span>
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

                    {/* Bottom Row: Stats & Session Info */}
                    <div className="w-full grid grid-cols-1 md:grid-cols-2 gap-8 mt-2">
                        {/* Stats Column */}
                        <div className="flex flex-col gap-6">
                            {/* Clinical Score Card */}
                            <div className="bg-surface-container-lowest p-6 rounded-lg shadow-[0_20px_40px_rgba(21,49,40,0.04)]">
                                <div className="flex justify-between items-center mb-6">
                                    <div>
                                        <p className="text-[10px] uppercase tracking-[0.1em] font-bold text-on-surface-variant mb-1">Clinical Score</p>
                                        <div className="flex items-baseline gap-3">
                                            <p className="text-3xl font-extrabold text-primary">
                                                {isScoring
                                                    ? <span className="animate-pulse text-on-surface-variant">Đang chấm...</span>
                                                    : clinicalScore !== null
                                                        ? Number(clinicalScore).toFixed(0)
                                                        : "--"}
                                                <span className="text-sm font-medium text-on-surface-variant ml-1">/50</span>
                                            </p>
                                            {/* Model Confidence Badge */}
                                            {clinicalScore !== null && !isScoring && (
                                                <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${confidence.color} text-white`}>
                                                    {confidence.label} confidence
                                                </span>
                                            )}
                                        </div>
                                        {/* Raw vs Calibrated score breakdown */}
                                        {clinicalScore !== null && rawScore !== null && !isScoring && (
                                            <p className="text-[9px] text-on-surface-variant mt-1">
                                                Raw: {Number(rawScore).toFixed(1)} → Calibrated: {Number(clinicalScore).toFixed(1)}
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
                                    <div className="h-full bg-[#6ABE4E] rounded-full transition-all duration-500" style={{ width: `${Math.min(100, (elapsedTime / 30) * 100)}%` }}></div>
                                </div>
                            </div>

                            {/* Secondary Actions */}
                            {isSaving && (
                                <div className="grid grid-cols-2 gap-3 pt-4">
                                    <button onClick={() => { }} className="py-3 px-2 bg-surface-container-high hover:bg-surface-variant rounded-lg text-xs font-bold text-on-surface-variant transition-colors flex flex-col items-center gap-1">
                                        <span className="material-symbols-outlined text-lg">pause</span> Pause
                                    </button>
                                    <button onClick={stopExerciseSession} className="py-3 px-2 bg-error-container/30 hover:bg-error-container text-error rounded-lg text-xs font-bold transition-colors flex flex-col items-center gap-1">
                                        <span className="material-symbols-outlined text-lg">stop</span> Stop
                                    </button>
                                </div>
                            )}
                        </div>

                        {/* Session Info Column */}
                        <div className="flex flex-col gap-6">
                            {/* Silhouette Guide Card (before exercise starts) */}
                            {!isSaving && !clinicalScore && !isScoring && (
                                <div className="bg-surface-container-low p-4 rounded-lg flex items-start gap-4">
                                    <span className="material-symbols-outlined text-primary mt-0.5">visibility</span>
                                    <div>
                                        <p className="font-bold text-primary text-sm">Position Guide</p>
                                        <p className="text-sm text-on-surface-variant mt-1">
                                            A green silhouette outline will appear on your camera feed.
                                            Match your body position to the outline for optimal exercise form.
                                        </p>
                                    </div>
                                </div>
                            )}

                            {/* Motion & Confidence Details (after scoring) */}
                            {clinicalScore !== null && (
                                <div className="p-4 rounded-lg flex items-start gap-4 border-l-4 bg-secondary-container/50 border-secondary">
                                    <span className="material-symbols-outlined text-secondary" style={{ fontVariationSettings: "'FILL' 1" }}>
                                        verified
                                    </span>
                                    <div className="flex-1">
                                        <p className="font-bold text-primary">Session Finished</p>
                                        <p className="text-sm text-on-surface-variant mt-1">Tuyệt vời! Điểm lâm sàng đã được ghi nhận.</p>

                                        {/* Confidence & Motion Details */}
                                        <div className="mt-3 space-y-2">
                                            {/* Model Confidence */}
                                            <div className="flex items-center gap-2">
                                                <span className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant w-20">Confidence</span>
                                                <div className={`w-2 h-2 rounded-full ${confidence.color}`}></div>
                                                <span className={`text-[11px] font-semibold ${confidence.textColor}`}>
                                                    {confidence.label}
                                                </span>
                                                {qualityFactor !== null && (
                                                    <span className="text-[10px] text-on-surface-variant ml-1">
                                                        (Q: {qualityFactor.toFixed(2)})
                                                    </span>
                                                )}
                                            </div>

                                            {/* Motion Energy Bar */}
                                            {motionEnergy !== null && (
                                                <div className="flex items-center gap-2">
                                                    <span className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant w-20">Motion</span>
                                                    <div className="h-1.5 w-20 bg-surface-container rounded-full overflow-hidden">
                                                        <div className={`h-full rounded-full transition-all ${motionEnergy > 0.5 ? 'bg-[#6ABE4E]' : motionEnergy > 0.2 ? 'bg-amber-400' : 'bg-red-400'
                                                            }`} style={{ width: `${Math.min(100, motionEnergy * 100)}%` }}></div>
                                                    </div>
                                                    <span className="text-[10px] text-on-surface-variant">{(motionEnergy * 100).toFixed(0)}%</span>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}