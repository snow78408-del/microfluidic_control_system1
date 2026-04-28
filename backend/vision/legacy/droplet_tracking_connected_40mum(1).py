"""
液滴追踪与磁珠计数（不考虑粘连/重叠版本）
- 基于 droplet_tracking_and_counting.py，但圆形检测不做「粘连分离」：
  每个通过圆度过滤的轮廓只当作一个圆，用轮廓质心+等效半径，不再用距离变换拆成多个圆。
- 可减少因“考虑粘连”而过滤掉的大量圆形。
"""

from droplet_tracking_and_counting import (
    DropletTracker,
    DropletTrack,
    TrackingResult,
    DropletBeads,
    DropletBeadsResult,
)
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import time


@dataclass
class ActiveDropletDetection:
    """单个 active 液滴的检测信息（圆心、半径、维护的磁珠个数）"""
    center: np.ndarray   # (x, y) 圆心
    radius: float
    droplet_id: int
    bead_count: int      # 当前维护的磁珠个数


def detect_and_track_beads_connected(
    active_detections: List[ActiveDropletDetection],
    edge_image: np.ndarray,
    base_image: np.ndarray,
    max_bead_area: int = 5,
) -> DropletBeadsResult:
    """
    基于 edge_image 的白色联通域检测磁珠，并更新液滴磁珠个数。

    逻辑：
    1. 找 edge_image 里所有白色联通域
    2. 对每个联通域：当 (1) 联通域内每个点都落在某个 active 液滴的圆内 (圆心+半径)
       (2) 面积 <= max_bead_area 时，才算一个磁珠
    3. 对每个 active 液滴，把属于它的磁珠个数加起来；若大于该液滴之前维护的磁珠数则更新

    Args:
        active_detections: 所有 active 液滴的 (圆心, 半径, droplet_id, 当前维护磁珠数)
        edge_image: 边缘二值图（白=255，黑=0），不是 black_regions
        base_image: 用于可视化的底图
        max_bead_area: 磁珠联通域最大面积（默认 5）

    Returns:
        DropletBeadsResult
    """
    if len(edge_image.shape) == 3:
        edge_image = cv2.cvtColor(edge_image, cv2.COLOR_BGR2GRAY)
    H, W = edge_image.shape[:2]

    # 白色 = 255，找所有白色联通域
    white_mask = (edge_image > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        white_mask, connectivity=8
    )

    # 每个 droplet_id -> 本帧检测到的磁珠个数；被计为磁珠的联通域 label；磁珠坐标列表
    droplet_new_count: Dict[int, int] = {d.droplet_id: 0 for d in active_detections}
    bead_labels: set = set()
    bead_positions_per_droplet: Dict[int, List[np.ndarray]] = {
        d.droplet_id: [] for d in active_detections
    }
    min_bead_area = 5
    max_bead_area = 25

    # 每个联通域：面积在范围内且所有点落在某液滴圆内 -> 算作该液滴的一个磁珠
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_bead_area or area > max_bead_area:
            continue
        ys, xs = np.where(labels == label)
        if len(xs) == 0:
            continue
        points = np.column_stack((xs, ys))  # (N, 2) 每行 (x, y)

        # 找「包含该联通域所有点」的液滴（圆心+半径的圆内）
        containing_droplet_id: Optional[int] = None
        for d in active_detections:
            cx, cy = float(d.center[0]), float(d.center[1])
            r = float(d.radius)
            dx = points[:, 0] - cx
            dy = points[:, 1] - cy
            if np.all(dx * dx + dy * dy < 0.81 * r * r):
                containing_droplet_id = d.droplet_id
                break
        if containing_droplet_id is not None and containing_droplet_id in droplet_new_count:
            droplet_new_count[containing_droplet_id] += 1
            bead_labels.add(label)
            cx_b, cy_b = centroids[label]
            bead_positions_per_droplet[containing_droplet_id].append(
                np.array([float(cx_b), float(cy_b)], dtype=np.float32)
            )

    # 构建结果：每个液滴 bead_count = max(之前维护的, 本帧检测到的)，并记录磁珠坐标
    droplets_beads: List[DropletBeads] = []
    total_beads = 0
    for d in active_detections:
        prev_count = d.bead_count
        new_count = droplet_new_count.get(d.droplet_id, 0)
        bead_count = max(prev_count, new_count)
        total_beads += bead_count
        beads_list = bead_positions_per_droplet.get(d.droplet_id, [])
        droplets_beads.append(
            DropletBeads(
                droplet_id=d.droplet_id,
                droplet_center=d.center.copy(),
                beads=beads_list,
                bead_count=bead_count,
            )
        )

    vis_image = base_image.copy()
    if len(vis_image.shape) == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
    edge_image_bgr = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR) if len(edge_image.shape) == 2 else edge_image.copy()
    raw_image = base_image.copy()
    if len(raw_image.shape) == 2:
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)

    # 把检测到的磁珠联通域用绿色显示（与 vis_image 同尺寸的 labels 区域）
    if vis_image.shape[:2] == labels.shape[:2] and bead_labels:
        bead_mask = np.isin(labels, list(bead_labels)).astype(np.uint8)
        vis_image[bead_mask > 0] = [0, 255, 0]

    for db in droplets_beads:
        if db.bead_count == 0:
            continue
        pt = (int(db.droplet_center[0]), int(db.droplet_center[1]))
        cv2.putText(vis_image, f"D{db.droplet_id}", (pt[0] + 15, pt[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(vis_image, f"Beads: {db.bead_count}", (pt[0] + 15, pt[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    vis_h = vis_image.shape[0]
    cv2.putText(vis_image, f"Total Beads: {total_beads}", (10, vis_h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(vis_image, f"Droplets: {len(active_detections)}", (10, vis_h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return DropletBeadsResult(
        droplets_beads=droplets_beads,
        vis_image=vis_image,
        raw_image=raw_image,
        edge_image=edge_image_bgr,
        total_beads=total_beads,
    )     


def detect_hollow_circle_centroids_no_overlap(
    gray_image: np.ndarray,
    min_radius: int = 19.30,
    max_radius: int = 27.50,
    min_dist_between_centers: int = 180,
    cut_line_ratio: float = 1.0,
    gaussian_blur_size: int = 5,
    min_contour_area: int = 500,
    circularity_threshold: float = 0.5,
    radius_min_ratio: float = 0.6,
    radius_max_ratio: float = 1.5,
    frame_index: Optional[int] = None,
    roi_x_start: Optional[int] = None,
    roi_x_end: Optional[int] = None,
    roi_y_start: Optional[int] = None,
    roi_y_end: Optional[int] = None,
    crop_top: Optional[int] = None,
):
    if len(gray_image.shape) == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

    # 归一化
    norm_u8 = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    H, W = norm_u8.shape

    # 磁珠专用：只做归一化，不做 CLAHE、不做任何模糊，保留小暗点清晰
    norm_u8_for_beads = norm_u8.copy()

    # 液滴轮廓：CLAHE + 保边平滑，用于得到光滑轮廓
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    norm_u8 = clahe.apply(norm_u8)
    norm_u8_smooth = cv2.bilateralFilter(norm_u8, d=5, sigmaColor=50, sigmaSpace=50)
    if gaussian_blur_size > 0:
        blur_size = gaussian_blur_size if gaussian_blur_size % 2 == 1 else gaussian_blur_size + 1
        norm_u8_smooth = cv2.GaussianBlur(norm_u8_smooth, (blur_size, blur_size), 0)

    # OTSU 得到液滴边缘二值图（轮廓细、边缘更光滑）
    _, binary = cv2.threshold(
        norm_u8_smooth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    edge_image = binary.copy()
    black_regions = 255 - edge_image

    # 磁珠：从未模糊的 norm_u8_for_beads 用百分位阈值，只保留最暗的一小部分（磁珠）
    # 这样磁珠不会被模糊抹掉，也不会和液滴内部大块白区合并
    bead_percentile = 12.0
    bead_dark_threshold = float(np.percentile(norm_u8_for_beads, bead_percentile))
    _, edge_image_for_beads = cv2.threshold(
        norm_u8_for_beads, bead_dark_threshold, 255, cv2.THRESH_BINARY_INV
    )
    cut_line = H * cut_line_ratio
    # =============================
    # 核心替换：白色联通域
    # =============================
    bw = (black_regions > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bw, connectivity=8
    )

    detections = []
    radii = []
    enclosing_radii = []
    detected_centers = []
    all_contour_info = []

    radius_min_threshold = 9.0
    radius_max_threshold = 25.0
    circularity_threshold = 1.0
    min_dist_between_centers = 40.80

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        cx, cy = centroids[label]

        component_mask = (labels == label).astype(np.uint8)

        # ========= 周长（像素级） =========
        eroded = cv2.erode(component_mask, kernel_erode)
        boundary = component_mask - eroded
        perimeter = float(boundary.sum())

        # ========= 半径 =========
        estimated_radius = np.sqrt(area / np.pi)

        ys, xs = np.where(component_mask > 0)
        if len(xs) > 0:
            dx = xs - cx
            dy = ys - cy
            enclosing_radius = float(np.sqrt(dx * dx + dy * dy).max())
        else:
            enclosing_radius = 0.0

        circularity = (
            4.0 * np.pi * area / (perimeter * perimeter)
            if perimeter > 0 else 0.0
        )

        # ========= 去重（提前算好，便于统一 DEBUG） =========
        is_duplicate = False
        # for px, py in detected_centers:
        #     if np.hypot(cx - px, cy - py) < min_dist_between_centers:
        #         is_duplicate = True
        #         break

        info = {
            "idx": label,
            "area": float(area),
            "center": (float(cx), float(cy)),
            "estimated_radius": float(estimated_radius),
            "enclosing_radius": float(enclosing_radius),
            "perimeter": float(perimeter),
            "circularity": float(circularity),
            "status": "",
        }

        # ========= 过滤逻辑（完全保留你的语义） =========
        if estimated_radius < radius_min_threshold or estimated_radius > radius_max_threshold:
            info["status"] = "半径过滤"
        elif cy > cut_line:
            info["status"] = "位置过滤"
        elif area < min_contour_area:
            info["status"] = "面积过滤"
        elif circularity < circularity_threshold:
            info["status"] = "圆度过滤"
        elif is_duplicate:
            info["status"] = "去重过滤"
        else:
            info["status"] = "✓ 通过"

        # DEBUG 打印当前联通域的全部检测信息（每一帧、每个 label 都打印）
        print(
            f"[Frame {frame_index}] label={label} | "
            f"center=({cx:.2f}, {cy:.2f}) | "
            f"area={area:.1f} | "
            f"perimeter={perimeter:.1f} | "
            f"estimated_radius={estimated_radius:.3f} | "
            f"enclosing_radius={enclosing_radius:.3f} | "
            f"circularity={circularity:.4f} | "
            f"半径范围=[{radius_min_threshold}, {radius_max_threshold}] | "
            f"cut_line={cut_line:.1f} | "
            f"min_contour_area={min_contour_area} | "
            f"circularity_threshold={circularity_threshold} | "
            f"status={info['status']}"
        )

        all_contour_info.append(info)

        if info["status"] != "✓ 通过":
            continue

        detections.append(np.array([cx, cy], dtype=np.float32))
        radii.append(float(estimated_radius))
        enclosing_radii.append(float(enclosing_radius))
        detected_centers.append((cx, cy))

    # 打印 estimated_radius DEBUG：每一帧都输出
    if frame_index is not None and radii:
        radii_str = ", ".join(f"{r:.2f}" for r in radii)
        print(f"[Frame {frame_index}] estimated_radius: {radii_str}")

    all_contour_centers = [c["center"] for c in all_contour_info]

    # 若传入 ROI 与 crop_top：在完成所有图形图像运算后，再裁剪出 edge_clean 和 black_regions，并得到裁剪区内的检测（局部坐标）
    if (
        roi_x_start is not None
        and roi_x_end is not None
        and roi_y_start is not None
        and roi_y_end is not None
        and crop_top is not None
    ):
        # 裁剪 edge / black_regions / edge_image_for_beads：先 ROI 再裁掉顶部 crop_top 行
        roi_edge = edge_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        edge_clean = roi_edge[crop_top:, :]
        roi_black = black_regions[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        black_regions = roi_black[crop_top:, :]
        roi_beads = edge_image_for_beads[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        edge_image_for_beads = roi_beads[crop_top:, :]

        # 裁剪区内边界（全图坐标）
        y_min = roi_y_start + crop_top
        y_max = roi_y_end
        x_min, x_max = roi_x_start, roi_x_end

        # 只保留圆心在裁剪区内的检测，并转为局部坐标
        detections_local = []
        radii_local = []
        centers_local = []
        for i, (det, r) in enumerate(zip(detections, radii)):
            cx, cy = float(det[0]), float(det[1])
            if x_min <= cx < x_max and y_min <= cy < y_max:
                detections_local.append(
                    np.array([cx - roi_x_start, cy - roi_y_start - crop_top], dtype=np.float32)
                )
                radii_local.append(r)
                centers_local.append((cx - roi_x_start, cy - roi_y_start - crop_top))

        all_contour_centers_local = []
        for (cx, cy) in all_contour_centers:
            if x_min <= cx < x_max and y_min <= cy < y_max:
                all_contour_centers_local.append(
                    (cx - roi_x_start, cy - roi_y_start - crop_top)
                )

        return detections_local, radii_local, edge_clean, black_regions, all_contour_centers_local, edge_image_for_beads

    return detections, radii, edge_image, black_regions, all_contour_centers, edge_image_for_beads

def process_video_no_overlap(
    video_path: str,
    output_path: Optional[str] = None,
    output_black_regions_path: Optional[str] = None,
    min_radius: int = 26,
    max_radius: int = 26,
    min_dist_between_centers: int = 40,
    circularity_threshold: float = 0.15,
    radius_min_ratio: float = 0.6,
    radius_max_ratio: float = 1.5,
):
    """处理视频：使用不处理粘连的圆形检测；可选单独保存 black_regions 视频。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 只要画面右边 3/4（左边大幅多放开）；去掉上边和下边各 1/5（保留中间 3/5）
    roi_x_start = int(width / 4)
    roi_x_end = width
    roi_width = roi_x_end - roi_x_start
    roi_y_start = int(height / 5)
    roi_y_end = int(height * 4 / 5)
    roi_height = roi_y_end - roi_y_start

    print(f"Video: {width}x{height} @ {fps} FPS (no-overlap mode)")
    print(f"ROI: {roi_width}x{roi_height}")

    writer = None
    writer_black = None
    video_w, video_h = None, None
    br_w, br_h = None, None
    fourcc = None
    droplet_tracker = DropletTracker(
        distance_threshold=90.0,
        max_unmatched_frames=7,
        inactive_top_margin=30,
    )
    frame_count = 0
    droplet_max_beads: Dict[int, int] = {}
    # 收集所有帧中检测到的液滴等效半径/面积，用于计算平均直径和平均面积
    all_detected_radii: List[float] = []
    all_detected_areas: List[float] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        start_time = time.time()

        # 先对全帧做图像处理，再在函数内按 ROI 裁剪
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop_height = (roi_y_end - roi_y_start) // 8

        detections, radii, edge_image, black_regions, all_contour_centers, edge_image_for_beads = detect_hollow_circle_centroids_no_overlap(
            gray_full,
            min_radius=min_radius,
            max_radius=max_radius,
            min_dist_between_centers=min_dist_between_centers,
            cut_line_ratio=1.0,
            gaussian_blur_size=3,
            min_contour_area=200,
            circularity_threshold=circularity_threshold,
            radius_min_ratio=radius_min_ratio,
            radius_max_ratio=radius_max_ratio,
            frame_index=frame_count,
            roi_x_start=roi_x_start,
            roi_x_end=roi_x_end,
            roi_y_start=roi_y_start,
            roi_y_end=roi_y_end,
            crop_top=crop_height,
        )

        # 与 edge_image / black_regions 同尺寸的裁剪帧（用于显示与磁珠检测）
        frame = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end][crop_height:, :]

        pos_to_radius = {}
        for det, r in zip(detections, radii):
            pos_to_radius[(int(det[0]), int(det[1]))] = r
            r_f = float(r)
            all_detected_radii.append(r_f)
            all_detected_areas.append(float(np.pi * r_f * r_f))

        track_result = droplet_tracker.update(detections)

        # 仅对 frame 70-86 和 125-137 打印 DEBUG
        _debug_frames = (70 <= frame_count <= 86) or (125 <= frame_count <= 137)
        if _debug_frames:
            # 打印 active droplet 信息：编号，上一帧坐标，unmatched_frames
            print(f"\n[Frame {frame_count}] Active droplets: {len(track_result.active_tracks)}")
            for track in track_result.active_tracks:
                prev = track.prev_position
                prev_str = f"({prev[0]:.1f}, {prev[1]:.1f})" if prev is not None else "N/A"
                print(f"  D{track.id}: 上一帧坐标={prev_str}, unmatched_frames={track.unmatched_frames}")

        # 构建 active detections：圆心、半径、droplet_id、当前维护磁珠数
        def _radius_for_track(track: DropletTrack) -> float:
            key = (int(round(track.position[0])), int(round(track.position[1])))
            if key in pos_to_radius:
                return pos_to_radius[key]
            if not pos_to_radius:
                return 50.0
            min_dist, best_r = float("inf"), 50.0
            for (px, py), r in pos_to_radius.items():
                d = np.hypot(track.position[0] - px, track.position[1] - py)
                if d < min_dist:
                    min_dist, best_r = d, r
            return best_r

        active_detections = [
            ActiveDropletDetection(
                center=track.position,
                radius=_radius_for_track(track),
                droplet_id=track.id,
                bead_count=droplet_max_beads.get(track.id, 0),
            )
            for track in track_result.active_tracks
        ]
        # 使用edge_image_for_beads检测磁珠：这是专门为磁珠检测创建的图像
        beads_result = detect_and_track_beads_connected(
            active_detections,
            edge_image_for_beads,  # 专门用于磁珠检测的图像，包含磁珠的白色区域
            frame,
            max_bead_area=5,
        )

        # 打印磁珠信息：属于哪一个 droplet，磁珠坐标（仅 frame 70-86 和 125-137）
        if _debug_frames:
            print(f"[Frame {frame_count}] Beads:")
            for db in beads_result.droplets_beads:
                for i, bead in enumerate(db.beads):
                    print(f"  droplet D{db.droplet_id}, 磁珠{i+1} 坐标=({bead[0]:.1f}, {bead[1]:.1f})")

        for det, radius in zip(detections, radii):
            center = (int(det[0]), int(det[1]))
            cv2.circle(beads_result.vis_image, center, int(radius), (255, 0, 0), 2)

        elapsed_ms = (time.time() - start_time) * 1000
        img_w = beads_result.vis_image.shape[1]
        text_x = img_w - 150
        cv2.putText(beads_result.vis_image, f"Frame: {frame_count}", (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(beads_result.vis_image, f"Time: {elapsed_ms:.1f}ms", (text_x, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(beads_result.vis_image, f"Detected: {len(detections)}", (text_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(beads_result.vis_image, f"New: {len(track_result.new_droplets)}", (text_x, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(beads_result.vis_image, f"Active: {len(track_result.active_tracks)}", (text_x, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(beads_result.vis_image, f"Total: {track_result.total_count}", (text_x, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        for db in beads_result.droplets_beads:
            prev_max = droplet_max_beads.get(db.droplet_id, 0)
            if db.bead_count > prev_max:
                droplet_max_beads[db.droplet_id] = db.bead_count

        actual_h, actual_w = beads_result.raw_image.shape[:2]
        if beads_result.edge_image.shape[:2] != (actual_h, actual_w):
            beads_result.edge_image = cv2.resize(beads_result.edge_image, (actual_w, actual_h))
        if beads_result.vis_image.shape[:2] != (actual_h, actual_w):
            beads_result.vis_image = cv2.resize(beads_result.vis_image, (actual_w, actual_h))
        if len(beads_result.raw_image.shape) == 2:
            beads_result.raw_image = cv2.cvtColor(beads_result.raw_image, cv2.COLOR_GRAY2BGR)
        if len(beads_result.edge_image.shape) == 2:
            beads_result.edge_image = cv2.cvtColor(beads_result.edge_image, cv2.COLOR_GRAY2BGR)
        if len(beads_result.vis_image.shape) == 2:
            beads_result.vis_image = cv2.cvtColor(beads_result.vis_image, cv2.COLOR_GRAY2BGR)

        # black_regions 并入输出画面：与 raw/edge 同尺寸，转 BGR，并在每个轮廓中心画紫色点
        black_regions_display = black_regions.copy()
        if black_regions_display.shape[:2] != (actual_h, actual_w):
            black_regions_display = cv2.resize(black_regions_display, (actual_w, actual_h))
        if len(black_regions_display.shape) == 2:
            black_regions_display = cv2.cvtColor(black_regions_display, cv2.COLOR_GRAY2BGR)
        purple_bgr = (255, 0, 255)
        for (cx, cy) in all_contour_centers:
            pt = (int(round(cx)), int(round(cy)))
            cv2.circle(black_regions_display, pt, 5, purple_bgr, 2)
            cv2.circle(black_regions_display, pt, 2, purple_bgr, -1)

        combined = np.hstack([
            beads_result.raw_image,
            beads_result.edge_image,
            black_regions_display,
            beads_result.vis_image,
        ])

        if output_path:
            combined_h, combined_w = combined.shape[:2]
            if frame_count == 1:
                video_w, video_h = combined_w, combined_h
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (video_w, video_h))
                if not writer.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (video_w, video_h))
                if writer and writer.isOpened():
                    print(f"Video writer: {output_path}, {video_w}x{video_h} @ {fps} FPS")
            if writer and writer.isOpened():
                if combined.shape[:2] != (video_h, video_w):
                    combined = cv2.resize(combined, (video_w, video_h))
                if combined.dtype != np.uint8:
                    combined = np.clip(combined, 0, 255).astype(np.uint8)
                if len(combined.shape) == 2:
                    combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
                writer.write(combined)

        # 单独保存 black_regions 视频（每帧为二值图，白=圆内/前景，黑=背景）
        if output_black_regions_path:
            bh, bw = black_regions.shape[:2]
            if frame_count == 1:
                br_w, br_h = bw, bh
                fourcc_br = cv2.VideoWriter_fourcc(*'XVID')
                writer_black = cv2.VideoWriter(output_black_regions_path, fourcc_br, fps, (br_w, br_h))
                if not writer_black.isOpened():
                    fourcc_br = cv2.VideoWriter_fourcc(*'MJPG')
                    writer_black = cv2.VideoWriter(output_black_regions_path, fourcc_br, fps, (br_w, br_h))
                if writer_black and writer_black.isOpened():
                    print(f"Black regions video: {output_black_regions_path}, {br_w}x{br_h} @ {fps} FPS")
            if writer_black and writer_black.isOpened():
                if black_regions.shape[:2] != (br_h, br_w):
                    black_regions = cv2.resize(black_regions, (br_w, br_h))
                frame_br = cv2.cvtColor(black_regions, cv2.COLOR_GRAY2BGR)
                if frame_br.dtype != np.uint8:
                    frame_br = np.clip(frame_br, 0, 255).astype(np.uint8)
                writer_black.write(frame_br)

        cv2.imshow("Raw | Edge | BlackRegions | Annotated (no overlap)", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer and writer.isOpened():
        writer.release()
    if writer_black and writer_black.isOpened():
        writer_black.release()
    cv2.destroyAllWindows()

    total_droplets = droplet_tracker.next_id - 1
    droplet_ids = list(range(1, total_droplets + 1))
    bead_counts = [droplet_max_beads.get(did, 0) for did in droplet_ids]
    empty_count = sum(1 for c in bead_counts if c == 0)
    single_count = sum(1 for c in bead_counts if c == 1)
    multi_count = sum(1 for c in bead_counts if c >= 2)
    empty_rate = empty_count / total_droplets * 100 if total_droplets > 0 else 0.0
    single_rate = single_count / total_droplets * 100 if total_droplets > 0 else 0.0
    multi_rate = multi_count / total_droplets * 100 if total_droplets > 0 else 0.0

    # 计算并输出液滴平均直径和平均面积（单位：像素 / 像素^2）
    if all_detected_radii:
        avg_radius = sum(all_detected_radii) / len(all_detected_radii)
        avg_diameter = 2.0 * avg_radius
        print(f"\nAverage droplet diameter (pixels): {avg_diameter:.2f}")
    if all_detected_areas:
        avg_area = sum(all_detected_areas) / len(all_detected_areas)
        print(f"Average droplet area (pixels^2): {avg_area:.2f}")

    print(f"\nProcessing complete. Total frames: {frame_count}")
    print(f"Total droplets tracked: {total_droplets}")
    print("===== Droplet bead statistics (no-overlap mode) =====")
    print(f"Empty (0 beads): {empty_count}  ({empty_rate:.2f}%)")
    print(f"Single (1 bead): {single_count}  ({single_rate:.2f}%)")
    print(f"Multi (>=2 beads): {multi_count}  ({multi_rate:.2f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="液滴追踪与磁珠计数（不处理粘连）")
    parser.add_argument("--video", type=str, help="输入视频路径")
    parser.add_argument("--output", type=str, default=None, help="输出视频路径（可选）")
    parser.add_argument("--output-black-regions", type=str, default=None, help="单独保存 black_regions 视频路径（可选）")
    parser.add_argument("--min-radius", type=int, default=26, help="最小圆半径")
    parser.add_argument("--max-radius", type=int, default=26, help="最大圆半径")
    parser.add_argument("--min-dist", type=int, default=40, help="圆心最小距离")
    parser.add_argument("--circularity", type=float, default=0.15, help="圆度阈值")
    parser.add_argument("--radius-min-ratio", type=float, default=0.6, help="等效半径下限比例")
    parser.add_argument("--radius-max-ratio", type=float, default=1.5, help="等效半径上限比例")
    args = parser.parse_args()

    if args.video:
        process_video_no_overlap(
            args.video,
            args.output,
            output_black_regions_path=args.output_black_regions,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            min_dist_between_centers=args.min_dist,
            circularity_threshold=args.circularity,
            radius_min_ratio=args.radius_min_ratio,
            radius_max_ratio=args.radius_max_ratio,
        )
    else:
        print("请提供 --video 输入视频路径")
        print("示例: python droplet_tracking_no_overlap.py --video tongji.mp4 [--output out.mp4]")
        print("      python droplet_tracking_no_overlap.py --video tongji.mp4 --output-black-regions black_regions.avi")
