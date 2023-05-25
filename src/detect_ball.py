"""
Finds the position of the ball given the MiRo's left and right undistorted camera feed
"""
import numpy as np  # Numerical Analysis library
import cv2
import os
import rospy

import actionlib
from cv_bridge import CvBridge, CvBridgeError  # ROS -> OpenCV converter
from joblib import load

from util.image_to_hog import get_hog_feature
from util.circle import Circle
from util.ball import Ball
from miro_football_perception.msg import AddDebugCameraAction, AddDebugCameraGoal, AddDebugBallAction, AddDebugBallGoal, AddDebugPitchBoxAction, AddDebugPitchBoxGoal
from util.save_images import ImageSaver
from kalman_filter import KalmanFilter1D
from miro_football_perception.srv import BatchBallTrajectoryPrediction


dir_path = os.path.dirname(os.path.realpath(__file__))


class DetectBall:
    DATASET_IMG_SIZE = (32, 32)
    PITCH_X = 1.95
    PITCH_Y = 1.3
    PITCH_BORDER = 0.1

    def __init__(self, image_to_world, debug=False):
        self.DEBUG = debug

        self.frames = {
            'l': {'frame': None},
            'r': {'frame': None}
        }

        # Initialise CV Bridge
        self.image_converter = CvBridge()

        self.image_to_world = image_to_world

        # list of possible circles
        self.circles = []
        # store for debugging display
        self.rejected_circles = []
        self.ball = None

        # used for collecting samples for training dataset
        self.image_saver = ImageSaver()

        self.classifier = load(dir_path + '/classifier/svm_classifier.gz')
        try:
            self.radius_to_range = load(dir_path + '/radius_range_correction.gz')
        except FileNotFoundError:
            # empirically calculated linear equation for simulation ball as a backup
            self.radius_to_range = lambda x: (33.8 - x) / 23

        self.kalman_filter = {
            'l': KalmanFilter1D(R=0.1, Q=5.1),
            'r': KalmanFilter1D(R=0.1, Q=5.1),
        }

        self.last_balls = np.empty((0, 2))
        self.max_last_balls = 10
        self.num_future_balls = 5

        rospy.wait_for_service('batch_get_ball_trajectory')
        self.predict_trajectory = rospy.ServiceProxy('batch_get_ball_trajectory', BatchBallTrajectoryPrediction)

        if self.DEBUG:
            self.debug_camera_client = actionlib.SimpleActionClient(
                'add_debug_camera', AddDebugCameraAction)
            self.debug_balls_client_accepted = actionlib.SimpleActionClient(
                'add_debug_ball_accepted', AddDebugBallAction)
            self.debug_balls_client_rejected = actionlib.SimpleActionClient(
                'add_debug_ball_rejected', AddDebugBallAction)
            self.debug_pitch_box_client = actionlib.SimpleActionClient('add_debug_pitch_box', AddDebugPitchBoxAction)

    # create a mask for white-ish pixels in the images
    def preprocess(self):
        for index in self.frames:
            frame = self.frames[index]['frame']

            # mask image from white/ grey
            sensitivity = 100
            hsv_lo_end = np.array([0, 0, 0])
            hsv_hi_end = np.array([180, sensitivity, 255])

            im_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

            mask = cv2.inRange(im_hsv, hsv_lo_end, hsv_hi_end)

            # mask image from black
            sensitivity = 30
            hsv_lo_end = np.array([0, 0, 0])
            hsv_hi_end = np.array([180, 255, sensitivity])

            im_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

            mask = mask | cv2.inRange(im_hsv, hsv_lo_end, hsv_hi_end)

            # Clean up the image
            seg = mask
            seg = cv2.GaussianBlur(seg, (5, 5), 0)
            seg = cv2.erode(seg, None, iterations=2)
            seg = cv2.dilate(seg, None, iterations=2)

            # store for later
            self.frames[index]['mask'] = mask
            self.frames[index]['clean_mask'] = seg

    # use the circular hough transform to find possible circles in the images
    def hough_circles(self):
        for index in self.frames:
            frame = self.frames[index]['clean_mask']

            # Fine-tune parameters
            ball_detect_min_dist_between_cens = 40  # Empirical
            canny_high_thresh = 10  # Empirical
            ball_detect_sensitivity = 10  # Lower detects more circles, so it's a trade-off
            # ball_detect_sensitivity = 35
            ball_detect_min_radius = 4  # Arbitrary, empirical
            ball_detect_max_radius = 40  # Arbitrary, empirical

            # Find circles using OpenCV routine
            # This function returns a list of circles, with their x, y and r values
            circles = cv2.HoughCircles(
                frame,
                cv2.HOUGH_GRADIENT,
                1,
                ball_detect_min_dist_between_cens,
                param1=canny_high_thresh,
                param2=ball_detect_sensitivity,
                minRadius=ball_detect_min_radius,
                maxRadius=ball_detect_max_radius,
            )

            if circles is not None:
                # convert circles to Circle objects
                for c in circles[0, :]:
                    self.circles.append(Circle(c[0], c[1], c[2], index=index))

    def apply_kalman_filter(self, tick):
        self.kalman_filter['l'].predict(tick)
        self.kalman_filter['r'].predict(tick)

        for c in self.circles:
            c.r = self.kalman_filter[c.stream_index].get_update(c.r)[0][0][0]

    def update_kalman_filter(self):
        for c in self.circles:
            self.kalman_filter[c.stream_index].update([[c.original_r]])
            c.r = self.kalman_filter[c.stream_index].get()[0]

    def stereo_depth(self):
        frame_l = cv2.cvtColor(self.frames['l']['frame'], cv2.COLOR_RGB2GRAY)
        frame_r = cv2.cvtColor(self.frames['r']['frame'], cv2.COLOR_RGB2GRAY)
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        stereo.setTextureThreshold(10)
        stereo.setMinDisparity(2)
        stereo.setNumDisparities(128)
        stereo.setBlockSize(15)
        stereo.setSpeckleRange(16)
        stereo.setSpeckleWindowSize(45)
        disparity = stereo.compute(frame_l, frame_r)
        return disparity

    # store images of the balls inside the Circle objects, used for debug display
    def save_circle_images(self):
        for c in self.circles:
            if c.stream_index is not None:
                frame = self.frames[c.stream_index]['frame']
                (x1, y1, x2, y2) = c.bbox(self.img_w, self.img_h)
                ball = frame[y1:y2, x1:x2]
                c.images.insert(0, ball)

    # mark a circle candidate as being rejected
    def reject_circle(self, c, rejected_by):
        c.rejected_by = rejected_by
        self.circles.remove(c)
        self.rejected_circles.append(c)

    # use k means clustering to reject circles that have too little white to be a valid ball
    def colour_filter(self):
        sensitivity = 60
        hsv_lo_end = np.array([0, 0, 0])
        hsv_hi_end = np.array([255, sensitivity, 255])

        for c in self.circles[:]:
            (x1, y1, x2, y2) = c.bbox(self.img_w, self.img_h)
            if c.stream_index is not None:
                # cut out only the ball from the image
                ball = self.frames[c.stream_index]['frame'][y1:y2, x1:x2]
                pixels = np.float32(ball.reshape(-1, 3))

                # k means clustering
                n_colors = 5
                if len(pixels) > n_colors:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
                    flags = cv2.KMEANS_RANDOM_CENTERS

                    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
                    _, counts = np.unique(labels, return_counts=True)

                    # if at least 10% of the image is grey then accept
                    # empirically calculated, this value is quite lenient, very few accurate balls should be rejected
                    min_percent_cover = 0.1
                    total = sum(counts)
                    in_range = False
                    # check if any of the clusters that make up 10% of the image are white-ish
                    for j, colour in enumerate(palette):
                        if counts[j] / total > min_percent_cover:
                            # convert to hsv
                            hsv = cv2.cvtColor(np.uint8([[colour]]), cv2.COLOR_BGR2HSV)
                            # mask agains white-ish range
                            mask = cv2.inRange(np.full((1, 1, 3), hsv, np.uint8), hsv_lo_end, hsv_hi_end)
                            if mask[0][0] > 0:
                                in_range = True
                                break
                    if not in_range:
                        self.reject_circle(c, 'colour')

    # classify candidates as either being a ball or not using its hog feature
    def hog_filter(self):
        for c in self.circles[:]:

            image = cv2.resize(
                c.images[0], self.DATASET_IMG_SIZE, interpolation=cv2.INTER_LINEAR)
            fd = get_hog_feature(image)

            predicted_class = self.classifier.predict(np.array([fd]))
            c.confidence = self.classifier.decision_function(np.array([fd]))
            if predicted_class == 0:
                self.reject_circle(c, 'hog')

    # if circle is too far away from previous then reject
    def distance_filter(self):
        box_to_draw = False
        future_balls = np.empty((0, 2))

        t = [t/10 for t in range(1, self.num_future_balls+1)]
        res = self.predict_trajectory(t)
        for (x, y) in zip(res.x, res.y):
            future_balls = np.append(future_balls, np.array([[x, y]]), axis=0)

        for c in self.circles[:]:
            [x, y] = self.image_to_world_space(c)
            if self.last_balls.shape[0] == self.max_last_balls:
                all_balls = np.append(self.last_balls, future_balls, axis=0)

                box_to_draw = True
                mean = np.mean(all_balls[:,0])
                std = max(np.std(all_balls[:,0]), 0.05)
                lower_x = mean - 3 * std
                upper_x = mean + 3 * std
                mean = np.mean(all_balls[:,1])
                std = max(np.std(all_balls[:,1]), 0.05)
                lower_y = mean - 3 * std
                upper_y = mean + 3 * std
                if not lower_x < x < upper_x or not lower_y < y < upper_y:
                    self.reject_circle(c, 'distance')
                elif not (-self.PITCH_X - self.PITCH_BORDER < x < self.PITCH_X + self.PITCH_BORDER
                    and -self.PITCH_Y - self.PITCH_BORDER < y < self.PITCH_Y + self.PITCH_BORDER):
                    self.reject_circle(c, 'within_pitch')
            
            self.last_balls = np.append(self.last_balls, np.array([[x, y]]), axis=0)
            if self.last_balls.shape[0] > self.max_last_balls:
                self.last_balls = np.delete(self.last_balls, 0, axis=0)
                
        if self.DEBUG and box_to_draw:
            goal = AddDebugPitchBoxGoal()

            goal.x1 = lower_x
            goal.y1 = lower_y
            goal.x2 = upper_x
            goal.y2 = upper_y

            self.debug_pitch_box_client.send_goal(goal)

    def best_per_frame_filter(self):
        for index in self.frames:
            frame_circles = list(filter(lambda c: c.stream_index == index, self.circles))
            if len(frame_circles) > 0:
                best_confidence = max(frame_circles, key=lambda c: c.confidence)
                best_index = frame_circles.index(best_confidence)
                for i, c in enumerate(frame_circles):
                    if i != best_index:
                        self.reject_circle(c, 'best_per_frame')

    # get the largest circle
    def get_biggest_circle(self):
        max_circle = None
        self.max_rad = 0
        for c in self.circles:
            if c.r > self.max_rad:
                self.max_rad = c.r
                max_circle = c
        return max_circle

    # convert a pixel position to the world position
    def image_to_world_space(self, circle):
        if circle.stream_index is not None:
            
            z = 0.04
            indices = ['l', 'r']
            ow1 = np.array(self.image_to_world.convert(
                circle.x, circle.y, 0, indices.index(circle.stream_index)))
            ow2 = np.array(self.image_to_world.convert(
                circle.x, circle.y, 5, indices.index(circle.stream_index)))

            v = ow2 - ow1
            v = self.image_to_world.correct(v, [circle.x, circle.y])
            # ow1 + t * v (3d line equation)
            # ow1[2] + t * v[2] = z (solve for t)
            t = (z - ow1[2]) / v[2]

            ow = ow1 + t * v
            return [ow[0], ow[1]]
        return None

    # store images to be manually classified and used in a dataset
    def save_image_for_dataset(self):
        for c in self.circles:
            self.image_saver.save_image(c.images[0], self.DATASET_IMG_SIZE)

    # find the best circle
    def filter_circles(self):
        self.save_circle_images()

        self.hog_filter()
        self.colour_filter()
        self.distance_filter()
        self.best_per_frame_filter()

        # self.save_image_for_dataset()

    # get the position of the best ball
    def get_ball_position(self):
        if self.ball is None:
            return None
        else:
            return self.image_to_world_space(self.ball)

    def get_ball_radius(self):
        if self.ball is None:
            return None
        else:
            return self.ball.r

    def get_ball_stream_index(self):
        if self.ball is None:
            return None
        else:
            return self.ball.stream_index

    def get_balls(self):
        balls = []
        for c in self.circles:
            balls.append(Ball(
                self.image_to_world_space(c),
                [c.x, c.y],
                c.r,
                c.stream_index,
                # calculate camera pos
                self.image_to_world.convert(0, 0, 0, ['l', 'r'].index(c.stream_index))
                ))
        return balls

    def debug_display(self):
        if self.DEBUG:

            for index in self.frames:
                # show frame with circles drawn on
                copy = self.frames[index]['frame'].copy()
                for c in self.circles:
                    if c.is_in_frame(index):
                        c.draw(copy)
                for c in self.rejected_circles:
                    if c.is_in_frame(index):
                        c.draw(copy, accepted=False)

                i = ['l', 'r'].index(index)

                goal = AddDebugCameraGoal()
                goal.x = 0
                goal.y = i
                goal.ros_image = self.image_converter.cv2_to_imgmsg(
                    copy, 'rgb8')

                self.debug_camera_client.send_goal(goal)

                # show masked image
                # mask_on_img = cv2.bitwise_and(self.frames[index]['frame'], cv2.cvtColor(self.frames[index]['mask'], cv2.COLOR_GRAY2BGR))
                mask_on_img = cv2.cvtColor(
                    self.frames[index]['clean_mask'], cv2.COLOR_GRAY2BGR)
                goal = AddDebugCameraGoal()
                goal.x = 1
                goal.y = i
                goal.ros_image = self.image_converter.cv2_to_imgmsg(
                    mask_on_img, 'rgb8')

                self.debug_camera_client.send_goal(goal)

            # show accepted and rejected balls
            for circles, client in [(self.circles, self.debug_balls_client_accepted), (self.rejected_circles, self.debug_balls_client_rejected)]:
                for c in circles:
                    for image in c.images:
                        goal = AddDebugBallGoal()
                        goal.ros_image = self.image_converter.cv2_to_imgmsg(
                            image, 'rgb8')

                        client.send_goal(goal)

            # DEPTH_VISUALIZATION_SCALE = 2048
            # cv2.imshow('depth', self.stereo_depth() / DEPTH_VISUALIZATION_SCALE)
            # cv2.waitKey(1)

    def reset(self):
        self.circles = []
        self.rejected_circles = []
        self.ball = None

    def detect_balls(self, frame_l, frame_r, tick=0.01):
        self.reset()

        self.img_h, self.img_w = frame_l.shape[:2]
        self.frames['l']['frame'] = frame_l
        self.frames['r']['frame'] = frame_r

        self.preprocess()
        self.hough_circles()
        self.apply_kalman_filter(tick)
        self.filter_circles()
        self.update_kalman_filter()
        self.debug_display()

        return self.get_balls()
