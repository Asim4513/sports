import argparse
from enum import Enum
from typing import Iterator, List

import math
import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = SoccerPitchConfiguration()

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


class Mode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'
    POSSESSION_TRACKING = 'POSSESSION_TRACKING'

def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.

    Args:
        players (sv.Detections): Detections of all players.
        players_team_id (np.array): Array containing team IDs of detected players.
        goalkeepers (sv.Detections): Detections of goalkeepers.

    Returns:
        np.ndarray: Array containing team IDs for the detected goalkeepers.

    This function calculates the centroids of the two teams based on the positions of
    the players. Then, it assigns each goalkeeper to the nearest team's centroid by
    calculating the distance between each goalkeeper and the centroids of the two teams.
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray
) -> np.ndarray:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    radar = draw_pitch(config=CONFIG)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]), radius=20, pitch=radar)
    return radar


def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels)
        yield annotated_frame


def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame


def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run ball detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        # overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame


def run_player_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player tracking on a video and yield annotated frames with tracked players.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels)
        yield annotated_frame


def run_team_classification(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run team classification on a video and yield annotated frames with team colors.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        yield annotated_frame


def run_radar(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels,
            custom_color_lookup=color_lookup)

        h, w, _ = frame.shape
        radar = render_radar(detections, keypoints, color_lookup)
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h
        )
        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
        yield annotated_frame


def run_possession_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player tracking, team classification, ball tracking, determine ball possession,
    and display live possession percentage stats. Yields annotated frames.
    """
    print("Initializing models and components for Possession Tracking...")
    # --- Initialization (Combine from team_classification and ball_detection) ---
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device) # Load ball model

    # Initialize Team Classifier and FIT IT FIRST (crucial!)
    print("Collecting initial crops for team classification...")
    team_classifier = TeamClassifier(device=device)
    initial_frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE) # Use stride to speed up crop collection
    initial_crops = []
    # Limit the number of frames used for fitting for speed, e.g., first 50 strided frames
    frame_count_for_fitting = 0
    max_frames_for_fitting = 50
    # Use try-except to handle potential StopIteration if video is shorter than max_frames_for_fitting * STRIDE
    try:
        with tqdm(initial_frame_generator, desc='Collecting crops for team fitting', total=max_frames_for_fitting) as pbar:
            for frame in pbar:
                if frame_count_for_fitting >= max_frames_for_fitting:
                    break
                result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(result)
                player_detections = detections[detections.class_id == PLAYER_CLASS_ID]
                if len(player_detections) > 0:
                    initial_crops.extend(get_crops(frame, player_detections))
                frame_count_for_fitting += 1
                # Manually update tqdm if total is known and iterator might be shorter
                # pbar.update(1) # Tqdm handles this automatically with iterable
    except StopIteration:
        print(f"\nReached end of video while collecting crops after {frame_count_for_fitting} frames.")

    if not initial_crops:
        print("Warning: No player crops collected for team classification fitting. Team IDs might be incorrect.")
    else:
        print(f"Fitting team classifier with {len(initial_crops)} crops...")
        team_classifier.fit(initial_crops)
    print("Team classifier fitting complete.")


    # Initialize Trackers
    player_tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    ball_tracker = BallTracker(buffer_size=20) # From run_ball_detection

    # Ball detection slicer (from run_ball_detection)
    def ball_callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)
    ball_slicer = sv.InferenceSlicer(callback=ball_callback, slice_wh=(640, 640))

    # Annotators
    player_ellipse_annotator = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(COLORS), thickness=2)
    player_label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(COLORS), text_color=sv.Color.WHITE, text_padding=5, text_thickness=1, text_position=sv.Position.BOTTOM_CENTER)
    ball_annotator = BallAnnotator(radius=10, thickness=2, buffer_size=10) # Customize ball annotation

    # --- Possession Statistics Variables --- START ---
    total_processed_frames = 0
    # Initialize counts for Team 0 and Team 1
    possession_counts = {0: 0, 1: 0}
    # --- Possession Statistics Variables --- END ---


    # --- Main Processing Loop ---
    print("Starting main frame processing loop...")
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    for frame in frame_generator: # frame_idx is not needed if using enumerate
        total_processed_frames += 1 # Increment total frame count
        annotated_frame = frame.copy()

        # --- Player Detection, Tracking, Team Classification ---
        player_result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        all_detections = sv.Detections.from_ultralytics(player_result)
        # Handle case where tracker might be empty initially or after losing tracks
        if len(all_detections) > 0:
            tracked_detections = player_tracker.update_with_detections(all_detections)
        else:
            # Ensure tracker is updated even with no detections to maintain internal state
            tracked_detections = player_tracker.update_with_detections(all_detections)


        # Filter detections by class
        players = tracked_detections[tracked_detections.class_id == PLAYER_CLASS_ID]
        goalkeepers = tracked_detections[tracked_detections.class_id == GOALKEEPER_CLASS_ID]
        referees = tracked_detections[tracked_detections.class_id == REFEREE_CLASS_ID]

        players_team_id = np.array([], dtype=int) # Default empty
        if len(players) > 0:
            player_crops = get_crops(frame, players)
            if player_crops: # Ensure crops are not empty before predicting
                players_team_id = team_classifier.predict(player_crops)
            else:
                 players_team_id = np.array([], dtype=int) # Handle no crops case


        goalkeepers_team_id = np.array([], dtype=int) # Default empty
        if len(goalkeepers) > 0 and len(players) > 0 and len(players_team_id) == len(players): # Ensure player IDs are valid for resolving GK team
             goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
        elif len(goalkeepers) > 0:
            goalkeepers_team_id = np.array([0] * len(goalkeepers)) # Simple fallback


        # Combine detections and create color lookup for annotation
        valid_detections_list = []
        team_ids_list = []

        if len(players) == len(players_team_id): # Check consistency before adding
            valid_detections_list.append(players)
            team_ids_list.extend(players_team_id.tolist())
        if len(goalkeepers) == len(goalkeepers_team_id): # Check consistency
            valid_detections_list.append(goalkeepers)
            team_ids_list.extend(goalkeepers_team_id.tolist())
        if len(referees) > 0:
            valid_detections_list.append(referees)
            team_ids_list.extend([REFEREE_CLASS_ID] * len(referees)) # Use referee class ID

        if not valid_detections_list:
            final_tracked_detections = sv.Detections.empty()
            color_lookup = np.array([], dtype=int)
            player_labels = []
        else:
            final_tracked_detections = sv.Detections.merge(valid_detections_list)
            color_lookup = np.array(team_ids_list)
            # Ensure tracker_id is present and matches length
            if hasattr(final_tracked_detections, 'tracker_id') and len(color_lookup) == len(final_tracked_detections.tracker_id):
                 player_labels = [f"T{tid if tid < 2 else 'R'} #{det_id}" # Label with Team (0/1) or R and track ID
                                  for tid, det_id in zip(color_lookup, final_tracked_detections.tracker_id)]
            else:
                 # Fallback if tracker_id is missing or length mismatch
                 player_labels = [f"T{tid if tid < 2 else 'R'}" for tid in color_lookup]


        # --- Ball Detection and Tracking ---
        ball_detections = ball_slicer(frame).with_nms(threshold=0.1)
        tracked_ball_detections = ball_tracker.update(ball_detections) # Use ball_tracker instance


        # --- Possession Logic ---
        player_in_possession_track_id = None
        team_in_possession_id = None # Reset each frame before calculation
        min_dist_to_ball = float('inf')

        if len(tracked_ball_detections) > 0 and len(final_tracked_detections) > 0 and hasattr(final_tracked_detections, 'tracker_id'):
            ball_xyxy = tracked_ball_detections.xyxy[0]
            ball_center_x = (ball_xyxy[0] + ball_xyxy[2]) / 2
            ball_center_y = (ball_xyxy[1] + ball_xyxy[3]) / 2
            ball_pos = np.array([ball_center_x, ball_center_y])

            player_anchors = final_tracked_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

            # Ensure color_lookup matches the number of detections being considered
            if len(color_lookup) == len(final_tracked_detections):
                relevant_indices = np.where(color_lookup != REFEREE_CLASS_ID)[0] # Indices of players/GKs
                if len(relevant_indices) > 0:
                    # Ensure player_anchors slice matches relevant_indices length
                    if len(player_anchors) == len(final_tracked_detections):
                        distances = np.linalg.norm(player_anchors[relevant_indices] - ball_pos, axis=1)
                        closest_player_local_idx = np.argmin(distances)
                        min_dist_to_ball = distances[closest_player_local_idx]
                        closest_player_global_idx = relevant_indices[closest_player_local_idx]

                        max_possession_distance = 150 # Example: pixels - ADJUST AS NEEDED

                        if min_dist_to_ball < max_possession_distance:
                            # Check if tracker_id exists and index is valid
                            if len(final_tracked_detections.tracker_id) > closest_player_global_idx:
                                player_in_possession_track_id = final_tracked_detections.tracker_id[closest_player_global_idx]
                                team_in_possession_id = color_lookup[closest_player_global_idx]

                                # --- Increment Possession Counter --- START ---
                                if team_in_possession_id in possession_counts:
                                    possession_counts[team_in_possession_id] += 1
                                # --- Increment Possession Counter --- END ---
                    else:
                        print(f"Warning: Mismatch between player_anchors ({len(player_anchors)}) and final_detections ({len(final_tracked_detections)}) length.")
            else:
                print(f"Warning: Mismatch between color_lookup ({len(color_lookup)}) and final_detections ({len(final_tracked_detections)}) length.")


        # --- Annotation ---
        # Annotate players/GKs/Referees
        if len(final_tracked_detections) > 0 and len(color_lookup) == len(final_tracked_detections):
            annotated_frame = player_ellipse_annotator.annotate(
                annotated_frame, final_tracked_detections, custom_color_lookup=color_lookup)
            # Ensure labels match detections before annotating
            if len(player_labels) == len(final_tracked_detections):
                 annotated_frame = player_label_annotator.annotate(
                    annotated_frame, final_tracked_detections, labels=player_labels, custom_color_lookup=color_lookup)


        # Annotate ball
        annotated_frame = ball_annotator.annotate(annotated_frame, tracked_ball_detections)

        # --- Annotate Possession Info & Statistics --- START ---
        possession_text = "Possession: None"
        if team_in_possession_id is not None:
            possession_text = f"Possession: Team {team_in_possession_id} (Player #{player_in_possession_track_id})"

        # Draw possession text
        frame_h, frame_w, _ = annotated_frame.shape
        # Slightly smaller font for stats maybe
        font_scale = 0.8
        font_thickness = 2
        text_y_start = 50

        cv2.putText(annotated_frame, possession_text, (50, text_y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # Calculate and draw possession percentages
        percentage_team_0 = (possession_counts[0] / total_processed_frames) * 100 if total_processed_frames > 0 else 0
        percentage_team_1 = (possession_counts[1] / total_processed_frames) * 100 if total_processed_frames > 0 else 0

        stats_text_0 = f"Team 0 Possession: {percentage_team_0:.1f}%"
        stats_text_1 = f"Team 1 Possession: {percentage_team_1:.1f}%"

        cv2.putText(annotated_frame, stats_text_0, (50, text_y_start + 40), # Position below
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, sv.Color.from_hex(COLORS[0]).as_bgr(), font_thickness, cv2.LINE_AA)
        cv2.putText(annotated_frame, stats_text_1, (50, text_y_start + 80), # Position below team 0
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, sv.Color.from_hex(COLORS[1]).as_bgr(), font_thickness, cv2.LINE_AA)
        # --- Annotate Possession Info & Statistics --- END ---


        yield annotated_frame # Yield the final annotated frame


def main(source_video_path: str, target_video_path: str, device: str, mode: Mode) -> None:
    if mode == Mode.PITCH_DETECTION:
        frame_generator = run_pitch_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.RADAR:
        frame_generator = run_radar(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.POSSESSION_TRACKING: # <-- ADD THIS BLOCK
        frame_generator = run_possession_tracking(
            source_video_path=source_video_path, device=device)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        with tqdm(total=video_info.total_frames, desc=f"Processing {mode.value}") as pbar:
            for frame in frame_generator:
                sink.write_frame(frame)
                pbar.update(1)

            cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mode', type=Mode, default=Mode.PLAYER_DETECTION)
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode
    )
