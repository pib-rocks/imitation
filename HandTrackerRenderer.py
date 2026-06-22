import cv2
import numpy as np

LINES_HAND = [[0,1],[1,2],[2,3],[3,4],
            [0,5],[5,6],[6,7],[7,8],
            [5,9],[9,10],[10,11],[11,12],
            [9,13],[13,14],[14,15],[15,16],
            [13,17],[17,18],[18,19],[19,20],[0,17]]

class HandTrackerRenderer:
    def __init__(self, tracker, output=None):
        self.tracker = tracker
        self.show_landmarks = True
        self.show_fps = True

        if output is None:
            self.output = None
        else:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.output = cv2.VideoWriter(
                output, fourcc, self.tracker.video_fps,
                (self.tracker.img_w, self.tracker.img_h)
            )

    def draw_hand(self, hand):
        if hand.lm_score <= self.tracker.lm_score_thresh:
            return

        thick_coef = hand.rect_w_a / 400
        lines = [
            np.array([hand.landmarks[point] for point in line]).astype(np.int32)
            for line in LINES_HAND
        ]
        cv2.polylines(self.frame, lines, False, (255, 0, 0), int(1 + thick_coef * 3), cv2.LINE_AA)
        radius = int(1 + thick_coef * 5)
        for x, y in hand.landmarks[:, :2]:
            cv2.circle(self.frame, (int(x), int(y)), radius, (0, 128, 255), -1)

    def draw(self, frame, hands, bag=None):
        self.frame = frame
        for hand in hands:
            self.draw_hand(hand)
        return self.frame

    def exit(self):
        if self.output:
            self.output.release()
        cv2.destroyAllWindows()

    def waitKey(self, delay=1):
        if self.show_fps:
            self.tracker.fps.draw(self.frame, orig=(50, 50), size=1, color=(240, 180, 100))
        cv2.imshow("Hand tracking", self.frame)
        if self.output:
            self.output.write(self.frame)
        key = cv2.waitKey(delay)
        if key == 32:
            key = cv2.waitKey(0)
            if key == ord('s'):
                cv2.imwrite("snapshot.jpg", self.frame)
        elif key == ord('f'):
            self.show_fps = not self.show_fps
        return key
