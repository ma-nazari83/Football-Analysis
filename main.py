from tracking import tracker
from team_assignment import team_assigner
from transformation import transformer
from visualization import p_renderer as renderer
import cv2
import supervision as sv
import pathlib
from transformation.football_coords import SoccerPitchConfiguration 
import numpy as np
from utils import video_utils

def main(video_path):
    all_frames = []
    pitch_frames =[]
    config = SoccerPitchConfiguration()
    t = tracker.Tracker()
    path = pathlib.Path(video_path)
    cap = cv2.VideoCapture(path)
    x = 0
    player_ids_and_teams = {}
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # if x==60:
        #     break
        # getting all objects  
        all_objects , detections = t.tracking(frame)
       
        # players_and_goalkeepers = [obj for obj in all_objects if obj.predicted_name in ['player' ]]

        playersss = detections[(detections["class_name"] =="player") ]

        # players_and_goalkeepers = detections[(detections["class_name"] =="player") | (detections["class_name"]=="goalkeeper")]
        # print(detections)

        # refrees = [obj for obj in all_objects if obj.predicted_name == 'refree']
        # ball = [obj for obj in all_objects if obj.predicted_name == 'ball']
        labels = team_assigner.my_assigning4(frame, playersss)
       
        # labels , sss = team_assigner.my_assigning4(frame, playersss)
        # for item in sss.keys():
        #     if item not in player_ids_and_teams.keys():
        #         player_ids_and_teams[int(item)] = sss[item]

        # for p in playersss:
        #     if p.tracker_id in player_ids_and_teams.keys():
        #         continue
        #     else:

        # detections[detections["class_name"]=="player" ]["team_label"] = labels
        # labels = [dataframe_with_labels[dataframe_with_labels["id"] == x.id]['team_label'] for x in all_objects]
        # print(detections)
        # print(player_ids_and_teams.keys())
        # labels = [player_ids_and_teams[int(x)] for x in detections[detections["class_name"]=="player" ].tracker_id]
        reference_frame_points ,keypoints , fil = transformer.pitch_detection(frame)
        real_points = np.array(config.vertices)[fil]
    
        new_labels = [7 if label == 0 else 8 for label in labels]
        # print("labels" , len(new_labels))
        c_a = sv.ColorAnnotator()

        ball_xy_on_pitch , player_xy_on_pitch , refree_xy_on_pitch, goaler_xy_on_pitch = renderer.transforming_view(detections,reference_frame_points , real_points)
        detections.class_id[detections["class_name"]=="player" ] = new_labels
        # print(detections[detections["class_name"]=="player" ].class_id)
        # print(type(detections[detections["class_name"]=="player" ]))
        #drawing the pitch
        pitch = renderer.draw_pitch(config=config)
        #ball on topdown view
        if ball_xy_on_pitch is not None:
            pitch = renderer.draw_points_on_pitch(
                config=config,
                xy= ball_xy_on_pitch.reshape(-1 ,2),
                face_color=sv.Color.WHITE,
                edge_color=sv.Color.BLACK,
                pitch=pitch
            )
        # print("playerrss" , player_xy_on_pitch.reshape(-1 ,2))
        #players team1 on topdown view #NEEDS TO BE ALTERED FOR THE COLOR
        pitch = renderer.draw_points_on_pitch(
            config=config,
            xy= player_xy_on_pitch.reshape(-1 ,2),
            face_color=sv.Color.RED,
            edge_color=sv.Color.BLACK,
            pitch=pitch,
            color_seq = labels

        )

        # #players team 2 on topdown view
        # pitch = renderer.draw_points_on_pitch(
        #     config=CONFIG,
        #     xy= player_xy_on_pitch[players_detection.class_id==2],
        #     face_color=sv.Color.BLUE,
        #     edge_color=sv.Color.BLACK,
        #     pitch=pitch
        # )

        #refree
        if refree_xy_on_pitch is not None:
            pitch = renderer.draw_points_on_pitch(
                config=config,
                xy= refree_xy_on_pitch.reshape(-1 ,2),
                face_color=sv.Color.YELLOW,
                edge_color=sv.Color.BLACK,
                pitch= pitch,
            )

        # print(sv.plot_image(pitch))
        # print()
        pitch_frames.append(pitch)
        # break

        # print(detections)
        bounding_box_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("000000") , thickness=2)
        my_palette = sv.ColorPalette.from_hex(["#FF0000", "#00FF00" , "#0000FF" , "#FFFF00" , "#000000"]) # 0=Red, 1=Green, 2=Blue, 3=Yellow
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER , color=my_palette , color_lookup=sv.ColorLookup.CLASS  , text_color=sv.Color.from_hex("000000") , text_padding = 2)
        annotated_frame = label_annotator.annotate(
            scene=frame.copy(),
            detections=detections,
            labels=detections.tracker_id
           
        )
      
        vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.from_hex('#FF1493'),
        radius=8)
        annotated_frame = vertex_annotator.annotate(
            scene=annotated_frame,
            key_points=keypoints)
        ellipse_annotator = sv.EllipseAnnotator(color_lookup=sv.ColorLookup.CLASS)
        annotated_frame = ellipse_annotator.annotate(
            scene=annotated_frame,
            detections = detections,
            
        )
        all_frames.append(annotated_frame)



       
        x+=1
    # Saving the annotated video
    # print("pitch frames" , pitch_frames)
    # print(len(pitch_frames))
    pitch_frames = np.array(pitch_frames)
    # print(pitch_frames.shape)
    video_utils.save_video(all_frames )
    video_utils.save_video(pitch_frames , "pitch.avi" , 1300 , 800)


if __name__ == "__main__":
    main(f"{DATA_DIR}/simple.mp4")
# main("/home/mohammad-amin/Desktop/footbal anaylsis/datasets/simple.mp4")

    