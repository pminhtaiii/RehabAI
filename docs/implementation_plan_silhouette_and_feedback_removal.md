# Implementation Plan: Silhouette Overlay + Remove Real-time Feedback

## [Overview]

Thêm reference pose silhouette overlay lên webcam canvas để hướng dẫn user đứng đúng vị trí exercise, đồng thời gỡ bỏ real-time feedback pipeline (backend WebSocket feedback handler + frontend feedback UI) để giảm latency và đơn giản hóa architecture.

**Vấn đề:**
1. Domain shift giữa Kinect training data và webcam inference — user không biết cần đứng ở đâu, tư thế nào
2. Real-time feedback tốn compute mỗi 30 frames nhưng Spearman thấp → không đáng tin cậy → UX xấu

**Lý do:**
- Silhouette overlay giải quyết camera framing consistency (giảm ~10-15% domain shift variance)
- Gỡ real-time feedback giảm latency, simplify codebase, tập trung vào clinical score endpoint

**Giải thích:**
- Reference CSV đã có sẵn trong `frontend/public/*.csv` với 17 keypoints × (y, x, confidence) cho ~520 frames
- Chỉ cần load 1 representative frame (hoặc interpolate mượt qua các frame) để vẽ silhouette
- Canvas rendering loop đã chạy 30fps → thêm silhouette draw là zero-additional-latency

**Trích nguồn:**
- MediaPipe PoseLandmarker docs: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
- Canvas2D API: https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D

---

## [Types]

Không có type system changes. Frontend sử dụng JavaScript/React, backend Python/FastAPI.

Dữ liệu reference keypoints format (từ CSV):
```
{
  nose_y, nose_x, nose_confidence,
  left_eye_y, left_eye_x, left_eye_confidence,
  right_eye_y, right_eye_x, right_eye_confidence,
  left_ear_y, left_ear_x, left_ear_confidence,
  right_ear_y, right_ear_x, right_ear_confidence,
  left_shoulder_y, left_shoulder_x, left_shoulder_confidence,
  right_shoulder_y, right_shoulder_x, right_shoulder_confidence,
  left_elbow_y, left_elbow_x, left_elbow_confidence,
  right_elbow_y, right_elbow_x, right_elbow_confidence,
  left_wrist_y, left_wrist_x, left_wrist_confidence,
  right_wrist_y, right_wrist_x, right_wrist_confidence,
  left_hip_y, left_hip_x, left_hip_confidence,
  right_hip_y, right_hip_x, right_hip_confidence,
  left_knee_y, left_knee_x, left_knee_confidence,
  right_knee_y, right_knee_x, right_knee_confidence,
  left_ankle_y, left_ankle_x, left_ankle_confidence,
  right_ankle_y, right_ankle_x, right_ankle_confidence
}
```

Reference pose data structure (mới, trong renderer):
```javascript
// Array of {x, y, confidence} cho 12 body landmarks
// Thứ tự khớp BODY_LANDMARKS indices: [11,12,13,14,15,16,23,24,25,26,27,28]
const referencePose = [
  { x: 0.5, y: 0.45, confidence: 0.9 },  // left_shoulder
  { x: 0.48, y: 0.45, confidence: 0.9 },  // right_shoulder
  // ... 10 more landmarks
];
```

---

## [Files]

### Files to modify:

1. **`frontend/src/pages/Exercise/renderer_canvas2d.js`** — Thêm silhouette drawing capability
   - Thêm method `drawSilhouette(referenceLandmarks, alpha)` 
   - Vẽ translucent body outline (connected segments + joint circles) dùng reference pose data
   - Thêm method `setReferencePose(referenceLandmarks)` để lưu reference pose

2. **`frontend/src/pages/Exercise/Exercise.jsx`** — Major refactor
   - **GỠ BỎ:** `compareJointsWithReference()` function (lines 253-283)
   - **GỠ BỎ:** `handleFeedbackResponse()` function (lines 217-241)
   - **GỠ BỎ:** `connectWebSocket()` feedback message handling (line 265-282 → only keep clinical_score connection)
   - **GỠ BỎ:** `buildKeypointArray()` function (lines 243-251)
   - **GỠ BỎ:** `currentFeedbackMessages` state, `feedbackMessages` state, `feedbackLatency` state
   - **GỠ BỎ:** Feedback UI section (lines 520-574) — "Real-time Guidance" panel
   - **THÊM:** Reference pose loading từ CSV → extract representative frame → set vào renderer
   - **THÊM:** Silhouette overlay rendering trong `detectPose()` loop
   - **GIỮ:** WebSocket connection nhưng chỉ cho `clinical_score` message type
   - **GIỮ:** Clinical score display UI

3. **`backend/main.py`** — Gỡ WebSocket feedback handler
   - **GỠ BỎ:** `FeedbackRequest` model (line 209-213) — không cần nữa vì REST feedback endpoint vẫn giữ
   - **GỠ BỎ:** `feedback` message type handling trong `exercise_session_ws()` (lines 624-640)
   - **GIỮ:** `clinical_score` message type handling trong WebSocket
   - **GIỮ:** REST API `POST /api/feedback/{exercise_id}` (lines 411-445) — có thể giữ cho future use

4. **`backend/feedback_engine.py`** — KHÔNG thay đổi (giữ nguyên, có thể dùng lại sau)

### Files to create:
- Không tạo file mới

### Files to delete:
- Không xóa file nào

---

## [Functions]

### New functions:

1. **`RendererCanvas2d.setReferencePose(landmarks)`** — `renderer_canvas2d.js`
   - Signature: `setReferencePose(landmarks: [{x, y, confidence}]) -> void`
   - Purpose: Lưu reference pose landmarks để vẽ silhouette

2. **`RendererCanvas2d.drawSilhouette(alpha)`** — `renderer_canvas2d.js`
   - Signature: `drawSilhouette(alpha: number = 0.2) -> void`
   - Purpose: Vẽ translucent reference pose silhouette lên canvas
   - Implementation: 
     - Vẽ connected segments (same connections as skeleton) với strokeStyle rgba
     - Vẽ joint circles với fillStyle rgba
     - Dùng Path2D cho performance (reuse across frames)
     - Pre-compute Path2D objects trong `setReferencePose()` để không phải tính lại mỗi frame

3. **`extractReferencePoseFromCSV(csvText, frameIndex)`** — `Exercise.jsx` (local function)
   - Signature: `extractReferencePoseFromCSV(csvText: string, frameIndex: number) -> [{x, y, confidence}]`
   - Purpose: Parse CSV text, extract 12 body landmarks từ frame chỉ định
   - Implementation: 
     - Parse CSV → lấy row `frameIndex` (default: frame ở giữa = row 260/520)
     - Map columns sang 12 body landmarks theo BODY_LANDMARKS order
     - Return array of {x, y, confidence}

### Modified functions:

1. **`detectPose()`** — `Exercise.jsx` (lines 120-188)
   - Change: Thêm `rendererRef.current.drawSilhouette(0.2)` sau khi draw video, trước khi draw user skeleton
   - Rendering order: video → silhouette (mờ) → user skeleton (rõ)

2. **`startExerciseSession()`** — `Exercise.jsx` (lines 336-367)
   - Change: Gỡ bỏ `connectWebSocket(exerciseInfo.exercise_id)` → chỉ connect WebSocket cho clinical_score khi cần
   - Gỡ bỏ reset của feedback-related states

3. **`stopExerciseSession()`** — `Exercise.jsx` (lines 369-378)
   - Change: Gỡ `disconnectWebSocket()` nếu WebSocket chỉ dùng cho clinical_score (hoặc giữ nếu dùng WS cho scoring)

4. **`exercise_session_ws()`** — `backend/main.py` (lines 601-693)
   - Change: Gỡ `feedback` message type handling, chỉ giữ `clinical_score` type

### Removed functions:

1. **`compareJointsWithReference()`** — `Exercise.jsx` (lines 253-283)
   - Reason: Real-time feedback không còn cần thiết
   - Migration: Không cần migration, gỡ hoàn toàn

2. **`handleFeedbackResponse()`** — `Exercise.jsx` (lines 217-241)
   - Reason: Không còn feedback response
   - Migration: Không cần migration

3. **`buildKeypointArray()`** — `Exercise.jsx` (lines 243-251)
   - Reason: Chỉ dùng cho feedback
   - Migration: Không cần migration

---

## [Classes]

### Modified classes:

1. **`RendererCanvas2d`** — `renderer_canvas2d.js`
   - Thêm property: `this.referenceLandmarks = null`
   - Thêm property: `this.silhouettePath = null` (pre-computed Path2D)
   - Thêm method: `setReferencePose(landmarks)`
   - Thêm method: `drawSilhouette(alpha)`
   - Constructor không thay đổi

---

## [Dependencies]

Không thêm dependency mới. Sử dụng Canvas2D API có sẵn (Path2D, rgba fill/stroke).

---

## [Testing]

### Manual testing approach:
1. Mở Exercise page → kiểm tra silhouette overlay hiển thị đúng vị trí
2. Kiểm tra silhouette mờ (alpha ~0.2) không che khuất user skeleton
3. Kiểm tra 30 giây exercise → clinical score vẫn hoạt động
4. Kiểm tra không có console errors từ feedback removal
5. Kiểm tra WebSocket vẫn connect/disconnect đúng cho clinical_score

### Validation:
- Silhouette phải khớp với reference video pose
- Không có latency increase (silhouette draw < 1ms per frame)
- Clinical score endpoint vẫn trả về đúng format

---

## [Implementation Order]

1. **Step 1:** Thêm silhouette methods vào `RendererCanvas2d` class
   - `setReferencePose()`, `drawSilhouette()`
   - Pre-compute Path2D cho performance

2. **Step 2:** Thêm `extractReferencePoseFromCSV()` trong `Exercise.jsx`
   - Parse reference CSV → extract representative frame
   - Set reference pose vào renderer sau khi CSV load xong

3. **Step 3:** Cập nhật `detectPose()` rendering loop
   - Thêm silhouette draw giữa video và user skeleton

4. **Step 4:** Gỡ real-time feedback khỏi frontend
   - Xóa `compareJointsWithReference`, `handleFeedbackResponse`, `buildKeypointArray`
   - Xóa feedback states: `currentFeedbackMessages`, `feedbackMessages`, `feedbackLatency`
   - Xóa feedback UI panel ("Real-time Guidance")
   - Simplify WebSocket: chỉ connect cho clinical_score

5. **Step 5:** Gỡ feedback handler khỏi backend WebSocket
   - Xóa `feedback` message type trong `exercise_session_ws()`
   - Giữ `clinical_score` type

6. **Step 6:** UI cleanup
   - Redesign feedback sidebar → thay bằng silhouette guide instructions
   - Giữ clinical score display, timer, recording indicator

7. **Step 7:** Test end-to-end
   - Verify silhouette renders correctly
   - Verify clinical score still works via WebSocket
   - Verify no console errors