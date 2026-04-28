"""
液滴追踪与磁珠计数系统
- 多目标追踪：基于最近邻匹配的液滴追踪
- 磁珠计数：检测每个液滴内的磁珠数量
"""

from numpy import ndarray
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional, Dict
import time


# =============================
# 数据结构定义
# =============================

@dataclass
class DropletTrack:
    """单个液滴的跟踪信息"""
    id: int                         # 液滴ID
    position: np.ndarray            # 当前位置 (x, y)
    unmatched_frames: int = 0       # 连续未匹配的帧数
    is_active: bool = True          # 是否活跃（未消失）
    prev_position: Optional[np.ndarray] = None  # 上一帧坐标（用于打印等）

    def __post_init__(self):
        self.position = np.array(self.position, dtype=np.float32)


@dataclass
class TrackingResult:
    """跟踪结果"""
    active_tracks: List[DropletTrack]                    # 活跃的跟踪轨迹
    matched_detections: List[Tuple[int, np.ndarray]]    # 匹配的检测结果 (track_id, position)
    new_droplets: List[np.ndarray]                       # 新出现的液滴位置
    total_count: int                                     # 累计检测到的液滴总数


@dataclass
class DropletBeads:
    """单个液滴内的磁珠信息"""
    droplet_id: int                  # 液滴ID
    droplet_center: np.ndarray       # 液滴中心位置
    beads: List[np.ndarray] = field(default_factory=list)  # 磁珠位置列表
    bead_count: int = 0              # 磁珠数量


@dataclass
class DropletBeadsResult:
    """所有液滴的磁珠检测结果"""
    droplets_beads: List[DropletBeads]   # 每个液滴的磁珠信息
    vis_image: np.ndarray                 # 可视化图像（带标注）
    raw_image: np.ndarray                 # 原始裁剪图像（不带标注）
    edge_image: np.ndarray                # Canny边缘检测图像
    total_beads: int                      # 所有液滴的磁珠总数


# =============================
# 液滴跟踪器类
# =============================

class DropletTracker:
    """
    多液滴跟踪器（基于最近邻匹配）
    
    使用示例:
        tracker = DropletTracker(distance_threshold=50.0, max_unmatched_frames=5)
        
        # 在每一帧中：
        detections = detect_droplets(frame)  # 检测液滴
        result = tracker.update(detections)   # 更新跟踪
        
        for track in result.active_tracks:
            print(f"液滴ID: {track.id}, 位置: {track.position}")
    """
    
    def __init__(
        self,
        distance_threshold: float = 300.0,
        max_unmatched_frames: int = 5,
        inactive_top_margin: Optional[int] = None,
    ):
        """
        初始化跟踪器
        
        # TODO: distance_threshold 和 max_unmatched_frames 可以根据实际情况调节。
        Args:
            distance_threshold: 匹配距离阈值（像素）  
            max_unmatched_frames: 最大未匹配帧数（超过此值认为液滴消失）
            inactive_top_margin: 圆心 y 小于此值时直接标为非 active（像素，None 表示不启用）
        """
        self.distance_threshold = distance_threshold
        self.max_unmatched_frames = max_unmatched_frames
        self.inactive_top_margin = inactive_top_margin
        self.tracks: List[DropletTrack] = []
        self.next_id = 1
    
    def update(self, detections: List[np.ndarray]) -> TrackingResult:
        """
        更新跟踪状态
        
        Args:
            detections: 当前帧检测到的液滴位置列表，每个元素为 (x, y)
        
        Returns:
            TrackingResult: 跟踪结果
        """
        result = TrackingResult(
            active_tracks=[],
            matched_detections=[],
            new_droplets=[],
            total_count=self.next_id - 1
        )
        
        # 如果当前帧没有检测到任何液滴
        if len(detections) == 0:
            for track in self.tracks:
                if track.is_active:
                    track.unmatched_frames += 1
                    if track.unmatched_frames > self.max_unmatched_frames:
                        track.is_active = False
                    elif (
                        self.inactive_top_margin is not None
                        and track.position[1] < self.inactive_top_margin
                    ):
                        track.is_active = False

            result.active_tracks = [t for t in self.tracks if t.is_active]
            result.total_count = self.next_id - 1
            return result
        
        # 转换检测结果为numpy数组
        detections = [np.array(d, dtype=np.float32) for d in detections]

        # 保存上一帧坐标（更新 position 前）
        for track in self.tracks:
            track.prev_position = track.position.copy()

        # 最近邻匹配
        matches = self._nearest_neighbor_matching(detections)
        
        # 标记匹配状态
        detection_matched = [False] * len(detections)
        track_matched = set()  # 使用集合存储已匹配的轨迹索引
        
        # 更新已匹配的轨迹
        for det_idx, track_idx in matches:
            detection_matched[det_idx] = True
            track_matched.add(track_idx)
            
            # 更新轨迹位置
            self.tracks[track_idx].position = detections[det_idx]
            self.tracks[track_idx].unmatched_frames = 0
            
            # 添加到匹配结果
            result.matched_detections.append(
                (self.tracks[track_idx].id, detections[det_idx])
            )
        
        # 先处理未匹配的轨迹：未匹配帧数+1（在添加新轨迹之前）
        num_existing_tracks = len(self.tracks)
        for i in range(num_existing_tracks):
            if self.tracks[i].is_active and i not in track_matched:
                self.tracks[i].unmatched_frames += 1
                if self.tracks[i].unmatched_frames > self.max_unmatched_frames:
                    self.tracks[i].is_active = False
        
        # 处理未匹配的检测结果：创建新轨迹
        for i, det in enumerate(detections):
            if not detection_matched[i]:
                # 创建新轨迹
                new_track = DropletTrack(id=self.next_id, position=det)
                self.tracks.append(new_track)
                self.next_id += 1
                result.new_droplets.append(det)
        
        # 圆心距画面上方小于 inactive_top_margin 时直接标为非 active
        if self.inactive_top_margin is not None:
            for track in self.tracks:
                if track.is_active and track.position[1] < self.inactive_top_margin:
                    track.is_active = False

        # 更新活跃轨迹列表
        result.active_tracks = [t for t in self.tracks if t.is_active]
        result.total_count = self.next_id - 1
        
        return result
    
    def _nearest_neighbor_matching(self, detections: List[np.ndarray]) -> List[Tuple[int, int]]:
        """
        最近邻匹配算法
        
        Returns:
            匹配对列表 [(detection_idx, track_idx), ...]
        """
        matches = []
        
        if len(detections) == 0 or len(self.tracks) == 0:
            return matches
        
        # 只考虑活跃的轨迹
        active_track_indices = [i for i, t in enumerate(self.tracks) if t.is_active]
        
        if len(active_track_indices) == 0:
            return matches
        
        # 贪心最近邻匹配
        track_used = [False] * len(active_track_indices)
        
        for det_idx, det in enumerate(detections):
            min_dist = self.distance_threshold
            best_track_idx = -1
            best_active_idx = -1
            
            # 找到最近的活跃轨迹（x和y方向的欧氏距离）
            for i, track_idx in enumerate(active_track_indices):
                if track_used[i]:
                    continue
                
                # 计算x和y方向的欧氏距离
                dist = np.linalg.norm(det - self.tracks[track_idx].position)
                
                if dist < min_dist:
                    min_dist = dist
                    best_track_idx = track_idx
                    best_active_idx = i
            
            # 如果找到匹配的轨迹
            if best_track_idx >= 0:
                matches.append((det_idx, best_track_idx))
                track_used[best_active_idx] = True
        
        return matches
    
    def get_active_tracks(self) -> List[DropletTrack]:
        """获取所有活跃的跟踪轨迹"""
        return [t for t in self.tracks if t.is_active]
    
    def reset(self):
        """重置跟踪器"""
        self.tracks = []
        self.next_id = 1


# =============================
# 液滴检测函数
# =============================

def detect_hollow_circle_centroids(
    gray_image: np.ndarray,
    min_radius: int = 80,
    max_radius: int = 90,
    circle_threshold: float = 30,
    min_dist_between_centers: int = 180,
    cut_line_ratio: float = 1.0,
    use_canny: bool = True,
    canny_low: float = 50,
    canny_high: float = 150,
    binary_threshold: int = 127,
    # 新增参数 - 用于处理双层圆环结构
    gaussian_blur_size: int = 5,
    dilate_iterations: int = 3,
    dilate_kernel_size: int = 7,
    min_contour_area: int = 500,
    circularity_threshold: float = 0.5,
    use_contour_method: bool = True,
    frame_index: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[float], np.ndarray]:
    """
    检测空心圆（仅边缘）的质心 - 改进版，支持双层圆环结构
    
    Args:
        gray_image: 灰度图像（uint8）
        min_radius: 最小圆半径
        max_radius: 最大圆半径
        circle_threshold: Hough圆检测的累加器阈值（越小检测越多但假，越大检测越少但准）
        min_dist_between_centers: 两个圆心之间的最小距离
        cut_line_ratio: 底部切割线比例（0.95表示底部5%区域忽略）
        use_canny: 是否使用Canny边缘检测
        canny_low: Canny低阈值
        canny_high: Canny高阈值
        binary_threshold: 二值化阈值（当use_canny=False时使用）
        gaussian_blur_size: 高斯模糊核大小（用于去噪，设为0禁用）
        dilate_iterations: 膨胀迭代次数（用于合并内外圈）
        dilate_kernel_size: 膨胀核大小
        min_contour_area: 最小轮廓面积（过滤小噪点）
        circularity_threshold: 圆度阈值（0-1，越接近1越圆）
        use_contour_method: 是否使用轮廓方法（推荐用于双层圆环）
        frame_index: 可选，当前帧号；传入时 DEBUG 输出会打印「第 x 帧」
    
    Returns:
        detections: 检测到的圆心位置列表
        radii: 检测到的圆半径列表
        edge_image: 处理后的边缘图像（用于调试）
    """
    # 确保是灰度图
    if len(gray_image.shape) == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    
    # 归一化
    norm_u8 = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    H, W = norm_u8.shape
    
    # ===== 1. 预处理：高斯模糊去噪 =====
    if gaussian_blur_size > 0:
        # 确保核大小为奇数
        blur_size = gaussian_blur_size if gaussian_blur_size % 2 == 1 else gaussian_blur_size + 1
        norm_u8 = cv2.GaussianBlur(norm_u8, (blur_size, blur_size), 0)
    
    # ===== 2. 直接阈值分割找黑色区域（不用Canny） =====
    # 圆内部是黑色的，直接用阈值找黑色区域
    # 使用自适应阈值或简单阈值
    
    # 方法：找到较暗的区域（圆的内部）
    # 先用OTSU自动确定阈值，或使用固定阈值
    _, binary = cv2.threshold(norm_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 开运算去除小噪点
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    
    # 用于显示的边缘图
    edge_image = binary_clean
    
    # 计算cut_line（基于原始高度，调用处会负责裁剪）
    cut_line = H
    
    detections = []
    radii = []
    
    if use_contour_method:
        # ===== 方法A：基于轮廓的检测，对粘连圆形使用距离变换分离 =====
        
        # edge_image是黑色区域变白色，我们需要找黑色区域（圆形内部）的轮廓
        # 反转：黑色区域（圆形内部）变成前景
        black_regions = 255 - edge_image  # 反转：黑色圆形内部变成白色
        
        # 先找所有轮廓
        contours, _ = cv2.findContours(black_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # 把图中每一块连通的白色区域的边界曲线提取出来；形状是[N, 1, 2]。轮廓有N个点，每个点坐标是(x, y)
        
        print(f"\n[DEBUG] ========== 圆形检测调试信息 ==========")
        if frame_index is not None:
            print(f"[DEBUG] 第 {frame_index} 帧")
        print(f"[DEBUG] 轮廓数量: {len(contours)}")
        print(f"[DEBUG] 当前参数: min_radius={min_radius}, max_radius={max_radius}")
        
        # 计算单个圆形的面积范围
        min_area = np.pi * (min_radius ** 2) * 0.99
        max_single_area = np.pi * (max_radius ** 2) * 1.01  # 单个圆形的最大面积
        max_enclosing_radius = max_radius * 1.0001  # 单个圆形的最小外接圆最大半径
        
        print(f"[DEBUG] 面积范围: min_area={min_area:.1f}, max_single_area={max_single_area:.1f}")
        print(f"[DEBUG] 半径范围: min_radius={min_radius}, max_radius={max_radius}, max_enclosing_radius={max_enclosing_radius:.1f}")
        
        # 计算距离变换（用于分离粘连的圆形）
        dist_transform = cv2.distanceTransform(black_regions, cv2.DIST_L2, 5)
        
        detected_centers = []
        all_candidate_info = []  # 存储所有候选圆的信息用于调试
        
        contour_idx = 0
        for contour in contours:
            contour_idx += 1
            area = cv2.contourArea(contour)
            (cx, cy), enclosing_radius = cv2.minEnclosingCircle(contour)
            estimated_radius = np.sqrt(area / np.pi)
            
            # 记录候选信息
            candidate_info = {
                'idx': contour_idx,
                'area': area,
                'enclosing_radius': enclosing_radius,
                'estimated_radius': estimated_radius,
                'center': (cx, cy),
                'status': 'unknown'
            }
            
            # 位置过滤
            if cy > cut_line:
                candidate_info['status'] = f'位置过滤: cy={cy:.1f} > cut_line={cut_line:.1f}'
                all_candidate_info.append(candidate_info)
                continue
            
            # 面积过滤：太小的轮廓直接跳过
            if area < min_area:
                candidate_info['status'] = f'面积太小: area={area:.1f} < min_area={min_area:.1f}'
                all_candidate_info.append(candidate_info)
                continue

            # 形状过滤：优先只保留“形状比较规则的圆”
            # 圆度 circularity = 4πA / P^2，完美圆 ≈ 1，越小越偏离圆形
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                candidate_info['status'] = f'周长异常: perimeter={perimeter:.2f}'
                all_candidate_info.append(candidate_info)
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            candidate_info['circularity'] = circularity
            if circularity < circularity_threshold:
                candidate_info['status'] = (
                    f'圆度过滤: circularity={circularity:.3f} < threshold={circularity_threshold:.3f}'
                )
                all_candidate_info.append(candidate_info)
                continue
            
            # 如果面积和半径都在正常范围内，直接使用轮廓质心
            if area <= max_single_area and enclosing_radius <= max_enclosing_radius:
                # 单个圆形，使用轮廓质心
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx_final = M["m10"] / M["m00"]
                    cy_final = M["m01"] / M["m00"]
                else:
                    cx_final, cy_final = cx, cy
                
                # 半径过滤
                radius_min_threshold = min_radius * 1.0
                radius_max_threshold = max_radius * 1.0
                if estimated_radius < radius_min_threshold or estimated_radius > radius_max_threshold:
                    candidate_info['status'] = f'半径过滤: estimated_radius={estimated_radius:.2f} 不在范围 [{radius_min_threshold:.2f}, {radius_max_threshold:.2f}]'
                    all_candidate_info.append(candidate_info)
                    continue
                
                # 去重
                is_duplicate = False
                for (prev_x, prev_y) in detected_centers:
                    dist_check = np.sqrt((cx_final - prev_x)**2 + (cy_final - prev_y)**2)
                    if dist_check < min_dist_between_centers:
                        is_duplicate = True
                        candidate_info['status'] = f'去重过滤: 距离已有中心太近 dist={dist_check:.1f} < {min_dist_between_centers}'
                        break
                
                if is_duplicate:
                    all_candidate_info.append(candidate_info)
                    continue
                
                # 通过所有过滤，添加到检测结果
                candidate_info['status'] = '✓ 检测到单个圆形'
                all_candidate_info.append(candidate_info)
                detections.append(np.array([cx_final, cy_final], dtype=np.float32))
                radii.append(float(estimated_radius))
                detected_centers.append((cx_final, cy_final))
            
            # 如果面积或半径过大，可能是多个圆形粘连，使用距离变换分离
            # elif area > max_single_area or enclosing_radius > max_enclosing_radius:
                candidate_info['status'] = f'可能粘连: area={area:.1f} > {max_single_area:.1f} 或 enclosing_radius={enclosing_radius:.2f} > {max_enclosing_radius:.2f}'
                all_candidate_info.append(candidate_info)
                
                # 创建该轮廓的mask
                contour_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, 255, -1)
                
                # 在该轮廓区域内，使用距离变换找局部最大值
                masked_dist = dist_transform.copy()
                masked_dist[contour_mask == 0] = 0
                
                # 使用较大的核来找到局部最大值（避免在单个圆内找到多个峰值）
                kernel_size = int(min_radius * 1.6)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                
                dilated = cv2.dilate(masked_dist, kernel, iterations=1)
                min_distance_value = min_radius * 0.5
                
                # 找到局部最大值
                local_peaks = (np.abs(masked_dist - dilated) < 1e-5) & (masked_dist > min_distance_value)
                local_peaks = local_peaks.astype(np.uint8) * 255
                
                # 连通域分析找到每个峰值
                local_num, local_labels, local_stats, local_centroids = cv2.connectedComponentsWithStats(local_peaks, 8, cv2.CV_32S)
                
                print(f"[DEBUG]   轮廓#{contour_idx}: 可能粘连，找到 {local_num-1} 个候选峰值")
                
                # 对每个峰值，验证并添加到检测结果
                for j in range(1, local_num):
                    local_cx, local_cy = local_centroids[j, 0], local_centroids[j, 1]
                    local_cx_int = int(round(local_cx))
                    local_cy_int = int(round(local_cy))
                    
                    if not (0 <= local_cy_int < H and 0 <= local_cx_int < W):
                        print(f"[DEBUG]      峰值#{j}: 坐标越界 ({local_cx_int}, {local_cy_int})")
                        continue
                    
                    local_radius = dist_transform[local_cy_int, local_cx_int]
                    
                    # 半径过滤：必须严格在范围内
                    radius_min_threshold = min_radius * 0.7
                    radius_max_threshold = max_radius * 1.2
                    if local_radius < radius_min_threshold or local_radius > radius_max_threshold:
                        print(f"[DEBUG]      峰值#{j}: 半径过滤 radius={local_radius:.2f} 不在范围 [{radius_min_threshold:.2f}, {radius_max_threshold:.2f}]")
                        continue
                    
                    # 验证：检查该中心周围是否有足够的黑色区域
                    mask_radius = int(local_radius * 1.2)
                    verify_mask = np.zeros((H, W), dtype=np.uint8)
                    cv2.circle(verify_mask, (local_cx_int, local_cy_int), mask_radius, 255, -1)
                    verify_mask = cv2.bitwise_and(verify_mask, contour_mask)
                    verify_area = np.sum(verify_mask > 0)
                    
                    # 验证面积应该在合理范围内
                    expected_area = np.pi * (local_radius ** 2)
                    if verify_area < expected_area * 0.4 or verify_area > expected_area * 2.0:
                        print(f"[DEBUG]      峰值#{j}: 面积验证失败 verify_area={verify_area:.1f}, expected={expected_area:.1f}")
                        continue
                    
                    # 去重
                    is_duplicate = False
                    for (prev_x, prev_y) in detected_centers:
                        dist_check = np.sqrt((local_cx - prev_x)**2 + (local_cy - prev_y)**2)
                        if dist_check < min_dist_between_centers:
                            is_duplicate = True
                            print(f"[DEBUG]      峰值#{j}: 去重过滤 距离已有中心太近 dist={dist_check:.1f}")
                            break
                    
                    if is_duplicate:
                        continue
                    
                    # 通过所有过滤，添加到检测结果
                    print(f"[DEBUG]      峰值#{j}: ✓ 检测到圆形 center=({local_cx:.1f}, {local_cy:.1f}), radius={local_radius:.2f}")
                    detections.append(np.array([local_cx, local_cy], dtype=np.float32))
                    radii.append(float(local_radius))
                    detected_centers.append((local_cx, local_cy))
        
        # 打印所有候选圆的统计信息
        print(f"\n[DEBUG] ========== 候选圆统计信息 ==========")
        print(f"[DEBUG] 总轮廓数: {len(contours)}")
        print(f"[DEBUG] 最终检测到: {len(detections)} 个圆")
        
        if len(radii) > 0:
            radii_array = np.array(radii)
            print(f"[DEBUG] 半径统计: min={radii_array.min():.2f}, max={radii_array.max():.2f}, mean={radii_array.mean():.2f}, median={np.median(radii_array):.2f}")
            print(f"[DEBUG] 半径分布:")
            hist, bins = np.histogram(radii_array, bins=10)
            for i in range(len(hist)):
                if hist[i] > 0:
                    print(f"[DEBUG]   [{bins[i]:.2f}, {bins[i+1]:.2f}]: {hist[i]} 个圆")
        
        # 打印前20个候选圆的详细信息
        print(f"\n[DEBUG] ========== 前20个候选圆详细信息 ==========")
        for info in all_candidate_info[:20]:
            print(f"[DEBUG] 轮廓#{info['idx']}: area={info['area']:.1f}, enclosing_r={info['enclosing_radius']:.2f}, "
                  f"estimated_r={info['estimated_radius']:.2f}, center=({info['center'][0]:.1f}, {info['center'][1]:.1f})")
            print(f"[DEBUG]  状态: {info['status']}")
        
        if len(all_candidate_info) > 20:
            print(f"[DEBUG] ... 还有 {len(all_candidate_info) - 20} 个候选圆未显示")
        
        print(f"[DEBUG] ==========================================\n")
        
        # 返回阈值图像用于调试
        edge_clean = edge_image
    
    else:
        # ===== 方法B：基于距离变换 + Hough的检测（原方法改进版） =====
        dist = cv2.distanceTransform(255 - edge_clean, cv2.DIST_L2, 5)
        
        # 动态阈值：基于膨胀后的厚度
        dist_threshold = max(5.0, dilate_kernel_size * dilate_iterations / 2)
        centerline = (dist > dist_threshold).astype(np.uint8) * 255
        
        # Hough圆检测
        circles = cv2.HoughCircles(
            centerline,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_dist_between_centers,
            param1=100,
            param2=circle_threshold,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                if y > cut_line:
                    continue
                detections.append(np.array([x, y], dtype=np.float32))
                radii.append(float(r))

    return detections, radii, edge_clean


# =============================
# 磁珠检测函数
# =============================

def detect_beads_in_droplet(
    droplet_center: np.ndarray,
    diff_u8: np.ndarray,
    droplet_radius: int = 50,
    min_area: int = 100,
    max_area: int = 1000,
    inner_radius_ratio: float = 0.85,
    border_margin: int = 2,
) -> List[np.ndarray]:
    """
    检测单个液滴内的磁珠（只检测圆形液滴区域内的磁珠）
    
    Args:
        droplet_center: 液滴中心位置 (x, y)
        diff_u8: 帧差二值图像
        droplet_radius: 液滴的估计半径（用于定义ROI）
        min_area: 磁珠最小面积
        max_area: 磁珠最大面积
    
    Returns:
        beads: 磁珠位置列表（只包含在液滴圆形区域内的磁珠）
    """
    beads = []
    
    if diff_u8 is None or diff_u8.size == 0:
        return beads
    
    H, W = diff_u8.shape[:2]
    
    # 定义ROI区域（方形）
    cx, cy = int(round(droplet_center[0])), int(round(droplet_center[1]))
    
    roi_x = max(0, cx - droplet_radius)
    roi_y = max(0, cy - droplet_radius)
    roi_w = min(W - roi_x, droplet_radius * 2)
    roi_h = min(H - roi_y, droplet_radius * 2)
    
    if roi_w <= 0 or roi_h <= 0:
        return beads
    
    # 提取ROI
    roi_diff = diff_u8[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    
    # 创建圆形mask，确保只在液滴「黑圆内部」检测
    # 为了满足“白点不与外面的白色圆壳相交”，这里使用略小于 droplet_radius 的内圆
    inner_radius = max(1, int(droplet_radius * inner_radius_ratio))
    circle_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    # ROI内的圆心位置
    local_cx = cx - roi_x
    local_cy = cy - roi_y
    cv2.circle(circle_mask, (local_cx, local_cy), inner_radius, 255, -1)
    
    # 应用圆形mask
    roi_diff_masked = cv2.bitwise_and(roi_diff, roi_diff, mask=circle_mask)
    
    # 形态学操作
    th = roi_diff_masked.copy()
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_open)
    
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    th = cv2.dilate(th, kernel_dilate, iterations=1)
    
    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th, 8, cv2.CV_32S)
    
    # 提取磁珠位置
    for i in range(1, num_labels):  # 0 是背景
        area = stats[i, cv2.CC_STAT_AREA]
        
        # 面积筛选：“稍微大一点的白点”
        if (min_area + 10) <= area <= (max_area - 10):
            cx_local = centroids[i, 0]
            cy_local = centroids[i, 1]

            # 距离圆心的距离（在 ROI 坐标系下）
            dist_to_center_local = np.sqrt(
                (cx_local - local_cx) ** 2 + (cy_local - local_cy) ** 2
            )

            # 要求：
            # 1. 在内圆之内（不靠近外壳）
            # 2. 预留一个 border_margin 的安全距离，避免与壳相交
            if dist_to_center_local > (inner_radius - border_margin):
                continue
            
            # 转换为全局坐标
            cx_global = roi_x + cx_local
            cy_global = roi_y + cy_local
            
            beads.append(np.array([cx_global, cy_global], dtype=np.float32))
    
    return beads


def detect_and_track_beads_in_droplets(
    tracked_droplets: List[DropletTrack],
    current_gray: np.ndarray,
    base_image: np.ndarray,
    edge_image: np.ndarray,
    droplet_radius: int = 50,
    min_area: int = 100,
    max_area: int = 1000
) -> DropletBeadsResult:
    """
    检测所有液滴内的磁珠并追踪
    
    Args:
        tracked_droplets: 跟踪到的液滴列表
        current_gray: 当前帧灰度图（float32或uint8）
        prev_gray: 上一帧灰度图
        base_image: 基础图像（用于可视化）
        edge_image: Canny边缘检测图像
        droplet_radius: 液滴的估计半径
        min_area: 磁珠最小面积
        max_area: 磁珠最大面积
    
    Returns:
        DropletBeadsResult: 磁珠检测结果
    """
    # 将边缘图像转换为3通道BGR以便显示
    if len(edge_image.shape) == 2:
        edge_image_bgr = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)
    else:
        edge_image_bgr = edge_image.copy()
    
    result = DropletBeadsResult(
        droplets_beads=[],
        vis_image=base_image.copy(),
        raw_image=base_image.copy(),  # 保存原始裁剪图像（不带标注）
        edge_image=edge_image_bgr,    # 保存Canny边缘检测图像
        total_beads=0
    )
    
    # 磁珠检测不再使用“帧差”，只基于当前帧的亮点：
    # 1) 对当前灰度图做归一化 + 模糊
    # 2) 用 OTSU 自动阈值，得到“亮点为白，背景为黑”的二值图
    norm_u8 = cv2.normalize(current_gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    blur = cv2.GaussianBlur(norm_u8, (5, 5), 0)
    _, diff_u8 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 对每个液滴检测磁珠
    for droplet in tracked_droplets:
        droplet_beads = DropletBeads(
            droplet_id=droplet.id,
            droplet_center=droplet.position.copy()
        )
        
        # 检测该液滴内的磁珠
        beads = detect_beads_in_droplet(
            droplet.position,
            diff_u8,
            droplet_radius,
            min_area,
            max_area
        )
        
        droplet_beads.beads = beads
        droplet_beads.bead_count = len(beads)
        result.total_beads += droplet_beads.bead_count
        
        result.droplets_beads.append(droplet_beads)
        
        # 可视化
        droplet_pt = (int(droplet.position[0]), int(droplet.position[1]))
        
        # 绘制液滴中心（绿色圆圈）
        cv2.circle(result.vis_image, droplet_pt, 10, (0, 255, 0), 2)
        
        # 绘制液滴ID
        cv2.putText(result.vis_image, f"D{droplet.id}",
                   (droplet_pt[0] + 15, droplet_pt[1] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 绘制磁珠（红色圆圈）
        for bead in beads:
            bead_pt = (int(bead[0]), int(bead[1]))
            cv2.circle(result.vis_image, bead_pt, 5, (0, 0, 255), 2)
            cv2.circle(result.vis_image, bead_pt, 2, (0, 0, 255), -1)
        
        # 显示磁珠数量
        cv2.putText(result.vis_image, f"Beads: {droplet_beads.bead_count}",
                   (droplet_pt[0] + 15, droplet_pt[1] + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # 显示统计信息
    H = result.vis_image.shape[0]
    cv2.putText(result.vis_image, f"Total Beads: {result.total_beads}",
               (10, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(result.vis_image, f"Droplets: {len(tracked_droplets)}",
               (10, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return result


# =============================
# 完整处理流程
# =============================

def process_video(video_path: str, output_path: Optional[str] = None):
    """
    处理视频：检测液滴、追踪液滴、统计每个液滴内的磁珠数量
    输出视频包含两个并排画面：左边是原始裁剪图像，右边是带标注的图像
    
    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径（包含并排的原始和标注画面，可选）
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 计算ROI区域：从左边1/3到右边8/11，只取上面1/3的高度
    roi_x_start = int(width / 3)
    roi_x_end = int(width * 8 / 11)
    roi_width = roi_x_end - roi_x_start
    roi_y_start = 0 # int(height / 10)  # 上面1/10的不要了
    roi_y_end = int(height / 3)  # 只取上面1/3
    roi_height = roi_y_end - roi_y_start
    
    print(f"Video: {width}x{height} @ {fps} FPS")
    print(f"ROI: x=[{roi_x_start}, {roi_x_end}], y=[{roi_y_start}, {roi_y_end}]")
    print(f"ROI size: {roi_width}x{roi_height}")
    
    # 创建视频写入器（可选）- 输出并排画面（宽度是ROI的三倍：原始 | Canny | 标注）
    writer = None
    video_w, video_h = None, None  # 将在第一帧时确定
    fourcc = None
    
    # 创建跟踪器
    droplet_tracker = DropletTracker(distance_threshold=90.0, max_unmatched_frames=5)
    
    frame_count = 0

    # 统计每个液滴在整个视频中的「最大磁珠数」
    # key: droplet_id, value: 该液滴在所有出现帧中的最大 bead_count
    droplet_max_beads: Dict[int, int] = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        start_time = time.time()
        
        # 裁剪ROI区域：水平方向从左边1/3到右边8/11，垂直方向只取上面1/10到上面1/3
        frame = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ===== 裁剪最上面1/8的画面（与detect_hollow_circle_centroids内部保持一致） =====
        H_orig, W_orig = gray.shape[:2]
        crop_height = H_orig // 8
        gray = gray[crop_height:, :]  # 去掉最上面1/8
        frame = frame[crop_height:, :]  # 同时裁剪frame，保持一致性
        
        # ===== 1. 检测液滴 =====
        # 参数说明：根据实际效果调整 dilate_iterations 和 dilate_kernel_size
        # - 边缘太粗：减小这两个参数
        # - 边缘太细/断裂：增大这两个参数
        detections, radii, edge_image = detect_hollow_circle_centroids(
            gray,
            min_radius=26,                 # 最小半径
            max_radius=26,                 # 最大半径
            circle_threshold=30,
            min_dist_between_centers=40,   # 进一步减小
            cut_line_ratio=1.0,            # 画面已裁剪为上1/3，检测全部
            use_canny=True,
            canny_low=50,
            canny_high=150,
            # 双层圆环处理参数（可调整）
            gaussian_blur_size=3,          # 高斯模糊（0=禁用）
            dilate_iterations=2,           # 闭运算迭代次数：1-3
            dilate_kernel_size=5,          # 闭运算核大小：3-7
            min_contour_area=200,          # 进一步减小
            frame_index=frame_count,       # DEBUG 用：标当前帧号
            circularity_threshold=0.15,    # 进一步降低
            use_contour_method=True # TODO: 根据实际情况调节
        )
        
        # 创建位置到半径的映射（用于打印）
        pos_to_radius = {}
        for det, r in zip(detections, radii):
            pos_to_radius[(int(det[0]), int(det[1]))] = r
        
        # ===== 2. 追踪液滴 =====
        track_result = droplet_tracker.update(detections)
        
        # ===== 3. 检测每个液滴内的磁珠 =====
        beads_result = detect_and_track_beads_in_droplets(
            track_result.active_tracks,
            gray.astype(np.float32),
            frame,
            edge_image,
            droplet_radius=50,
            min_area=100,
            max_area=1000
        )

        # 更新每个液滴的最大磁珠数（用于后续统计单包 / 空包 / 多包）
        for db in beads_result.droplets_beads:
            prev_max = droplet_max_beads.get(db.droplet_id, 0)
            if db.bead_count > prev_max:
                droplet_max_beads[db.droplet_id] = db.bead_count
        
        # ===== 4. 绘制检测到的圆（蓝色圆圈） =====
        for i, (det, radius) in enumerate(zip(detections, radii)):
            center = (int(det[0]), int(det[1]))
            cv2.circle(beads_result.vis_image, center, int(radius), (255, 0, 0), 2)
        
        # ===== 5. 显示处理时间和统计信息（右侧） =====
        elapsed_ms = (time.time() - start_time) * 1000
        img_w = beads_result.vis_image.shape[1]
        text_x = img_w - 150  # 右侧位置
        
        cv2.putText(beads_result.vis_image, f"Frame: {frame_count}",
                   (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(beads_result.vis_image, f"Time: {elapsed_ms:.1f}ms",
                   (text_x, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 显示本帧检测和跟踪信息
        cv2.putText(beads_result.vis_image, f"Detected: {len(detections)}",
                   (text_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(beads_result.vis_image, f"New: {len(track_result.new_droplets)}",
                   (text_x, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(beads_result.vis_image, f"Active: {len(track_result.active_tracks)}",
                   (text_x, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(beads_result.vis_image, f"Total: {track_result.total_count}",
                   (text_x, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # ===== 6. 打印每个液滴的信息（包括半径） =====
        if frame_count % 30 == 0:  # 每30帧打印一次
            print(f"\nFrame {frame_count}:")
            print(f"  Active Droplets: {len(track_result.active_tracks)}")
            print(f"  Total Beads: {beads_result.total_beads}")
            print(f"  Detected circles: {len(detections)}")
            for i, (det, r) in enumerate(zip(detections, radii)):
                print(f"    Circle {i+1}: center=({det[0]:.1f}, {det[1]:.1f}), radius={r:.1f}")
            for db in beads_result.droplets_beads:
                # 尝试找到对应的半径
                pos_key = (int(db.droplet_center[0]), int(db.droplet_center[1]))
                radius_str = ""
                # 查找最近的检测点的半径
                min_dist = float('inf')
                matched_radius = None
                for (px, py), r in pos_to_radius.items():
                    dist = ((px - pos_key[0])**2 + (py - pos_key[1])**2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        matched_radius = r
                if matched_radius is not None and min_dist < 50:
                    radius_str = f", radius={matched_radius:.1f}"
                print(f"    Droplet {db.droplet_id}: {db.bead_count} beads at ({db.droplet_center[0]:.1f}, {db.droplet_center[1]:.1f}){radius_str}")
        
        # 合并画面：原始裁剪图像 | Canny边缘图像 | 带标注图像
        # 确保所有图像具有相同的尺寸
        actual_h, actual_w = beads_result.raw_image.shape[:2]
        
        # 确保edge_image和vis_image与raw_image尺寸一致
        if beads_result.edge_image.shape[:2] != (actual_h, actual_w):
            beads_result.edge_image = cv2.resize(beads_result.edge_image, (actual_w, actual_h))
        if beads_result.vis_image.shape[:2] != (actual_h, actual_w):
            beads_result.vis_image = cv2.resize(beads_result.vis_image, (actual_w, actual_h))
        
        # 确保所有图像都是3通道BGR格式
        if len(beads_result.raw_image.shape) == 2:
            beads_result.raw_image = cv2.cvtColor(beads_result.raw_image, cv2.COLOR_GRAY2BGR)
        if len(beads_result.edge_image.shape) == 2:
            beads_result.edge_image = cv2.cvtColor(beads_result.edge_image, cv2.COLOR_GRAY2BGR)
        if len(beads_result.vis_image.shape) == 2:
            beads_result.vis_image = cv2.cvtColor(beads_result.vis_image, cv2.COLOR_GRAY2BGR)
        
        combined = np.hstack([beads_result.raw_image, beads_result.edge_image, beads_result.vis_image])
        
        # 写入视频（并排画面）
        if output_path:
            combined_h, combined_w = combined.shape[:2]
            
            # 在第一帧时创建VideoWriter（使用实际帧尺寸）
            if frame_count == 1:
                video_w, video_h = combined_w, combined_h
                # 尝试使用XVID编码器
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (video_w, video_h))
                if not writer.isOpened():
                    print(f"Warning: XVID codec failed. Trying MJPG...")
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (video_w, video_h))
                    if not writer.isOpened():
                        print(f"Warning: MJPG codec failed. Trying mp4v...")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(output_path, fourcc, fps, (video_w, video_h))
                        if not writer.isOpened():
                            print(f"Error: Cannot create video writer for {output_path}")
                            print(f"  Tried codecs: XVID, MJPG, mp4v")
                            print(f"  Frame dimensions: {video_w}x{video_h}")
                            writer = None
                else:
                    print(f"Video writer created: {output_path}, {video_w}x{video_h} @ {fps} FPS")
            
            if writer:
                # 确保combined帧的尺寸与VideoWriter匹配
                if combined.shape[:2] != (video_h, video_w):
                    combined = cv2.resize(combined, (video_w, video_h))
                
                # 验证帧格式（必须是uint8的BGR图像）
                if combined.dtype != np.uint8:
                    combined = np.clip(combined, 0, 255).astype(np.uint8)
                
                # 确保是3通道BGR格式
                if len(combined.shape) == 2:
                    combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
                elif combined.shape[2] != 3:
                    # 如果是4通道（BGRA），转换为3通道
                    combined = cv2.cvtColor(combined, cv2.COLOR_BGRA2BGR)
                
                # 写入帧
                success = writer.write(combined)
                if not success and frame_count % 30 == 0:  # 每30帧报告一次错误
                    print(f"Warning: Failed to write frame {frame_count}")
                    print(f"  Frame shape: {combined.shape}, dtype: {combined.dtype}")
                    print(f"  Expected size: {video_w}x{video_h}")
        
        # 显示三个并排画面
        cv2.imshow("Raw | Canny Edge | Annotated", combined)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    total_droplets = droplet_tracker.next_id - 1

    # 如果某个液滴从未在 droplet_max_beads 中出现，视为 0 个磁珠
    droplet_ids = list(range(1, total_droplets + 1))
    bead_counts = [droplet_max_beads.get(did, 0) for did in droplet_ids]

    # 这里采用微流控常规定义：
    # - 空包: 0 个磁珠
    # - 单包: 恰好 1 个磁珠
    # - 多包: >= 2 个磁珠
    empty_count = sum(1 for c in bead_counts if c == 0)
    single_count = sum(1 for c in bead_counts if c == 1)
    multi_count = sum(1 for c in bead_counts if c >= 2)

    empty_rate = empty_count / total_droplets * 100 if total_droplets > 0 else 0.0
    single_rate = single_count / total_droplets * 100 if total_droplets > 0 else 0.0
    multi_rate = multi_count / total_droplets * 100 if total_droplets > 0 else 0.0
    
    print(f"\nProcessing complete. Total frames: {frame_count}")
    print(f"Total droplets tracked: {total_droplets}")
    print("===== Droplet bead statistics (per droplet, using max bead count) =====")
    print(f"Empty droplets (0 beads): {empty_count}  ({empty_rate:.2f}%)")
    print(f"Single droplets (1 bead): {single_count}  ({single_rate:.2f}%)")
    print(f"Multi droplets (>=2 beads): {multi_count}  ({multi_rate:.2f}%)")


# =============================
# 单帧处理函数（便于集成）
# =============================

class DropletBeadCounter:
    """
    液滴磁珠计数器（封装完整处理流程）
    
    使用示例:
        counter = DropletBeadCounter()
        
        # 在每一帧中：
        result = counter.process_frame(frame)
        
        # 访问结果
        for db in result.droplets_beads:
            print(f"液滴{db.droplet_id}: {db.bead_count}个磁珠")
        
        cv2.imshow("Result", result.vis_image)
    """
    
    def __init__(
        self,
        distance_threshold: float = 50.0,
        max_unmatched_frames: int = 5,
        droplet_min_radius: int = 25,
        droplet_max_radius: int = 26,
        circle_threshold: float = 30,
        min_dist_between_centers: int = 80,
        droplet_radius_for_beads: int = 50,
        bead_min_area: int = 100,
        bead_max_area: int = 1000,
        use_canny: bool = True,
        canny_low: float = 50,
        canny_high: float = 150,
        roi_left_ratio: float = 1/3,
        roi_right_ratio: float = 5/7,
        # 新增参数 - 处理双层圆环结构
        gaussian_blur_size: int = 3,
        dilate_iterations: int = 2,
        dilate_kernel_size: int = 5,
        min_contour_area: int = 500,
        circularity_threshold: float = 0.4,
        use_contour_method: bool = True
    ):
        """
        初始化计数器
        
        Args:
            distance_threshold: 液滴跟踪距离阈值
            max_unmatched_frames: 最大未匹配帧数
            droplet_min_radius: 液滴最小半径
            droplet_max_radius: 液滴最大半径
            circle_threshold: Hough圆检测阈值
            min_dist_between_centers: 圆心最小距离
            droplet_radius_for_beads: 用于磁珠检测的液滴半径
            bead_min_area: 磁珠最小面积
            bead_max_area: 磁珠最大面积
            use_canny: 是否使用Canny边缘检测
            canny_low: Canny低阈值
            canny_high: Canny高阈值
            roi_left_ratio: ROI左边界比例（默认1/3）
            roi_right_ratio: ROI右边界比例（默认5/7）
            gaussian_blur_size: 高斯模糊核大小（去噪）
            dilate_iterations: 膨胀迭代次数（合并内外圈）
            dilate_kernel_size: 膨胀核大小
            min_contour_area: 最小轮廓面积（过滤噪点）
            circularity_threshold: 圆度阈值（0-1）
            use_contour_method: 是否使用轮廓方法
        """
        self.tracker = DropletTracker(distance_threshold, max_unmatched_frames)
        
        self.droplet_min_radius = droplet_min_radius
        self.droplet_max_radius = droplet_max_radius
        self.circle_threshold = circle_threshold
        self.min_dist_between_centers = min_dist_between_centers
        self.droplet_radius_for_beads = droplet_radius_for_beads
        self.bead_min_area = bead_min_area
        self.bead_max_area = bead_max_area
        self.use_canny = use_canny
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.roi_left_ratio = roi_left_ratio
        self.roi_right_ratio = roi_right_ratio
        
        # 新增参数
        self.gaussian_blur_size = gaussian_blur_size
        self.dilate_iterations = dilate_iterations
        self.dilate_kernel_size = dilate_kernel_size
        self.min_contour_area = min_contour_area
        self.circularity_threshold = circularity_threshold
        self.use_contour_method = use_contour_method
        
        # 不再需要上一帧灰度图
        self.frame_count = 0
    
    def process_frame(self, frame: np.ndarray) -> DropletBeadsResult:
        """
        处理单帧图像
        
        Args:
            frame: BGR图像
        
        Returns:
            DropletBeadsResult: 磁珠检测结果
        """
        self.frame_count += 1
        
        # 裁剪ROI区域：从左边roi_left_ratio到右边roi_right_ratio
        h, w = frame.shape[:2]
        roi_x_start = int(w * self.roi_left_ratio)
        roi_x_end = int(w * self.roi_right_ratio)
        frame = frame[:, roi_x_start:roi_x_end]
        
        # 转换为灰度图
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # 1. 检测液滴
        detections, radii, edge_image = detect_hollow_circle_centroids(
            gray,
            min_radius=self.droplet_min_radius,
            max_radius=self.droplet_max_radius,
            circle_threshold=self.circle_threshold,
            min_dist_between_centers=self.min_dist_between_centers,
            use_canny=self.use_canny,
            canny_low=self.canny_low,
            canny_high=self.canny_high,
            gaussian_blur_size=self.gaussian_blur_size,
            dilate_iterations=self.dilate_iterations,
            dilate_kernel_size=self.dilate_kernel_size,
            min_contour_area=self.min_contour_area,
            circularity_threshold=self.circularity_threshold,
            use_contour_method=self.use_contour_method,
            frame_index=self.frame_count,
        )
        
        # 2. 追踪液滴
        track_result = self.tracker.update(detections)
        
        # 3. 检测每个液滴内的磁珠
        beads_result = detect_and_track_beads_in_droplets(
            track_result.active_tracks,
            gray.astype(np.float32),
            frame if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR),
            edge_image,
            droplet_radius=self.droplet_radius_for_beads,
            min_area=self.bead_min_area,
            max_area=self.bead_max_area
        )
        
        # 绘制检测到的圆
        for det, radius in zip(detections, radii):
            center = (int(det[0]), int(det[1]))
            cv2.circle(beads_result.vis_image, center, int(radius), (255, 0, 0), 2)
        
        return beads_result
    
    def reset(self):
        """重置计数器"""
        self.tracker.reset()
        self.frame_count = 0
    
    def get_droplet_count(self) -> int:
        """获取累计液滴数量"""
        return self.tracker.next_id - 1


# =============================
# 主函数
# =============================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="液滴追踪与磁珠计数")
    parser.add_argument("--video", type=str, help="输入视频路径")
    parser.add_argument("--output", type=str, default=None, help="输出视频路径（包含并排的原始和标注画面，可选）")
    parser.add_argument("--camera", type=int, default=None, help="摄像头ID（如果不使用视频文件）")
    
    args = parser.parse_args()
    
    if args.video:
        process_video(args.video, args.output)
    elif args.camera is not None:
        # 使用摄像头
        cap = cv2.VideoCapture(args.camera)
        counter = DropletBeadCounter()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = counter.process_frame(frame)
            # 三个画面并排显示：原始 | Canny边缘 | 标注
            combined = np.hstack([result.raw_image, result.edge_image, result.vis_image])
            cv2.imshow("Raw | Canny Edge | Annotated", combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("请提供视频路径 (--video) 或摄像头ID (--camera)")
        print("\n使用示例:")
        print("  python droplet_tracking_and_counting.py --video input.mp4")
        print("  python droplet_tracking_and_counting.py --video input.mp4 --output output.mp4")
        print("  python droplet_tracking_and_counting.py --camera 0")
        print("\n输出视频包含三个并排画面：")
        print("  左边：原始裁剪图像（不带标注）")
        print("  中间：Canny边缘检测图像")
        print("  右边：带标注的图像（液滴ID、磁珠位置等）")
        print("\n或者在Python中直接使用:")
        print("  from droplet_tracking_and_counting import DropletBeadCounter")
        print("  counter = DropletBeadCounter()")
        print("  result = counter.process_frame(frame)")
        print("  # result.vis_image: 带标注的图像")
        print("  # result.raw_image: 原始裁剪图像（不带标注）")
        print("  # result.edge_image: Canny边缘检测图像")
