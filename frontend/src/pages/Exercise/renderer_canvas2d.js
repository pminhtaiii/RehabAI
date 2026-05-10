// Canvas renderer for MediaPipe PoseLandmarker results.
// Draws body keypoints and skeleton connections.
// Includes silhouette overlay for reference pose guidance.

const DEFAULT_LINE_WIDTH = 2;
const DEFAULT_RADIUS = 4;
const SCORE_THRESHOLD = 0.3;

// Silhouette styling
const SILHOUETTE_COLOR = 'rgba(106, 190, 78, ';  // Green with alpha channel
const SILHOUETTE_LINE_WIDTH = 3;
const SILHOUETTE_JOINT_RADIUS = 6;

// MediaPipe Pose body connections (index-based, 33 landmarks)
const POSE_CONNECTIONS = [
    [11, 12], // shoulders
    [11, 13], [13, 15], // left arm
    [12, 14], [14, 16], // right arm
    [11, 23], [12, 24], // torso
    [23, 24], // hips
    [23, 25], [25, 27], // left leg
    [24, 26], [26, 28], // right leg
];

// Mapping from BODY_LANDMARKS order (12 landmarks) to MediaPipe indices (33 landmarks).
// BODY_LANDMARKS order: left_shoulder(11), right_shoulder(12), left_elbow(13), right_elbow(14),
//   left_wrist(15), right_wrist(16), left_hip(23), right_hip(24), left_knee(25), right_knee(26),
//   left_ankle(27), right_ankle(28)
const BODY_TO_MEDIAPIPE = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];

const LEFT_INDICES = [11, 13, 15, 23, 25, 27];
const RIGHT_INDICES = [12, 14, 16, 24, 26, 28];

export class RendererCanvas2d {
    constructor(canvas) {
        this.ctx = canvas.getContext('2d');
        this.videoWidth = canvas.width;
        this.videoHeight = canvas.height;

        // Silhouette reference pose storage
        this.referenceLandmarks = null;  // Array of {x, y, confidence} for 33 MediaPipe indices
        this._silhouetteSegmentsPath = null;  // Pre-computed Path2D for skeleton segments
        this._silhouetteJointsPath = null;    // Pre-computed Path2D for joint circles
    }

    /**
     * Set reference pose landmarks for silhouette overlay.
     * @param {Array<{x: number, y: number, confidence: number}>} landmarks
     *   Array of 12 body landmarks in BODY_LANDMARKS order.
     *   Will be remapped to full 33-index MediaPipe array internally.
     */
    setReferencePose(landmarks) {
        if (!landmarks || landmarks.length !== 12) {
            console.warn('[Renderer] setReferencePose: expected 12 landmarks, got', landmarks?.length);
            return;
        }

        // Map 12 body landmarks → full 33-index array (fill null for unused indices)
        const fullLandmarks = new Array(33).fill(null);
        for (let i = 0; i < 12; i++) {
            const mpIndex = BODY_TO_MEDIAPIPE[i];
            fullLandmarks[mpIndex] = landmarks[i];
        }
        this.referenceLandmarks = fullLandmarks;

        // Pre-compute Path2D objects for performance (reuse across frames)
        this._precomputeSilhouettePaths();
    }

    /**
     * Pre-compute Path2D for silhouette skeleton segments and joint circles.
     * Called once when reference pose is set — avoids per-frame Path2D allocation.
     */
    _precomputeSilhouettePaths() {
        if (!this.referenceLandmarks) return;

        const lm = this.referenceLandmarks;
        const w = this.videoWidth;
        const h = this.videoHeight;

        // Build skeleton segments Path2D
        const segPath = new Path2D();
        for (const [i, j] of POSE_CONNECTIONS) {
            const kp1 = lm[i];
            const kp2 = lm[j];
            if (!kp1 || !kp2) continue;
            if ((kp1.confidence ?? 1) < SCORE_THRESHOLD || (kp2.confidence ?? 1) < SCORE_THRESHOLD) continue;

            segPath.moveTo(kp1.x * w, kp1.y * h);
            segPath.lineTo(kp2.x * w, kp2.y * h);
        }
        this._silhouetteSegmentsPath = segPath;

        // Build joint circles Path2D (using arc sub-paths)
        const jointPath = new Path2D();
        for (let i = 0; i < 33; i++) {
            const kp = lm[i];
            if (!kp) continue;
            if ((kp.confidence ?? 1) < SCORE_THRESHOLD) continue;

            jointPath.moveTo(kp.x * w + SILHOUETTE_JOINT_RADIUS, kp.y * h);
            jointPath.arc(kp.x * w, kp.y * h, SILHOUETTE_JOINT_RADIUS, 0, 2 * Math.PI);
        }
        this._silhouetteJointsPath = jointPath;
    }

    /**
     * Draw translucent silhouette overlay of reference pose.
     * Should be called after drawing video frame but before drawing user skeleton.
     * @param {number} alpha - Opacity [0, 1], default 0.2
     */
    drawSilhouette(alpha = 0.2) {
        if (!this.referenceLandmarks || !this._silhouetteSegmentsPath) return;

        const ctx = this.ctx;
        ctx.save();

        // Draw skeleton segments
        ctx.strokeStyle = SILHOUETTE_COLOR + alpha + ')';
        ctx.lineWidth = SILHOUETTE_LINE_WIDTH;
        ctx.lineCap = 'round';
        ctx.stroke(this._silhouetteSegmentsPath);

        // Draw joint circles
        ctx.fillStyle = SILHOUETTE_COLOR + alpha + ')';
        ctx.fill(this._silhouetteJointsPath);

        ctx.restore();
    }

    /**
     * Invalidate pre-computed silhouette paths when canvas size changes.
     * Call this after updating videoWidth/videoHeight.
     */
    invalidateSilhouetteCache() {
        if (this.referenceLandmarks) {
            this._precomputeSilhouettePaths();
        }
    }

    drawResults(landmarks) {
        if (!landmarks || landmarks.length === 0) return;
        const lm = landmarks[0];
        this.drawKeypoints(lm);
        this.drawSkeleton(lm);
    }

    drawKeypoints(landmarks) {
        this.ctx.lineWidth = DEFAULT_LINE_WIDTH;

        for (let i = 0; i < landmarks.length; i++) {
            const kp = landmarks[i];
            const score = kp.visibility ?? 1;
            if (score < SCORE_THRESHOLD) continue;

            if (LEFT_INDICES.includes(i)) {
                this.ctx.fillStyle = 'Green';
            } else if (RIGHT_INDICES.includes(i)) {
                this.ctx.fillStyle = 'Orange';
            } else {
                this.ctx.fillStyle = 'Red';
            }
            this.ctx.strokeStyle = 'White';

            const circle = new Path2D();
            circle.arc(kp.x * this.videoWidth, kp.y * this.videoHeight, DEFAULT_RADIUS, 0, 2 * Math.PI);
            this.ctx.fill(circle);
            this.ctx.stroke(circle);
        }
    }

    drawSkeleton(landmarks) {
        this.ctx.strokeStyle = 'White';
        this.ctx.lineWidth = DEFAULT_LINE_WIDTH;

        for (const [i, j] of POSE_CONNECTIONS) {
            const kp1 = landmarks[i];
            const kp2 = landmarks[j];
            if (!kp1 || !kp2) continue;

            const s1 = kp1.visibility ?? 1;
            const s2 = kp2.visibility ?? 1;
            if (s1 < SCORE_THRESHOLD || s2 < SCORE_THRESHOLD) continue;

            this.ctx.beginPath();
            this.ctx.moveTo(kp1.x * this.videoWidth, kp1.y * this.videoHeight);
            this.ctx.lineTo(kp2.x * this.videoWidth, kp2.y * this.videoHeight);
            this.ctx.stroke();
        }
    }
}