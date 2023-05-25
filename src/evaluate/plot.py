"""
Plot data on a 2D or 3D graph using the observations recorded by the system
"""

from httplib2 import RelativeURIError
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
import sys
from joblib import load

BALL_R = 0.04

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = DIR_PATH + '/observation_data.gz'

MIRO_POSE_X = 0
MIRO_POSE_Y = 1
MIRO_POSE_THETA = 2
REAL_POS_X = 3
REAL_POS_Y = 4
PERCEPT_POS_X = 5
PERCEPT_POS_Y = 6
IMAGE_POS_X = 7
IMAGE_POS_Y = 8
RADIUS = 9
STREAM_INDEX = 10
TIME = 11
FRAME = 12
REAL_VEL_X = 13
REAL_VEL_Y = 14
PERCEPT_VEL_X = 15
PERCEPT_VEL_Y = 16
CAMERA_POS_X = 17
CAMERA_POS_Y = 18
CAMERA_POS_Z = 19
HEAD_LIFT = 20
TAG = 21


help = """Program for plotting data recorded frm the ball vision system

The first argument is the index of the data to plot.
Do `python3 plot.py data` to get the shape of all data

The following variables can be plotted:
- frame
- time
- percept_x
- percept_y
- absolute_error
- relative_error
- percent_error
- percept_range
- percept_range_camera
- real_range
- real_range_camera
- range_error
- range_error_camera
- image_x
- image_y
- radius
- percept_azimuth
- real_azimuth
- azimuth_error
- percept_elevation
- real_elevation
- elevation_error
- percept_velocity_direction
- real_velocity_direction
- velocity_direction_error
- percept_velocity_length
- real_velocity_length
- velocity_length_error
- head_lift

Variables can be altered with these commands: ( do -[command] e.g -deg )
- deg: convert data to degrees
- lt[num]: only include data less than num
- gt[num]: only include data greater than num

To select which data to include: ( do -[command] e.g. -left )
- left (l): only data from the left camera
- right (r): only data from the right camera
- observe (o): data from the left and right cameras
- estimate (e): the best estimate of the system (using observations and kalman filter)

To change the colour of a selection: ( e.g. -#red)
- #[colour] to change the colour of the plot

Example:
python3 plot.py 280 image_x azimuth_error -deg -lt30 -gt-30 -l -#cyan -r -#blue

"""


# functions for retrieving data 
def frame(data):
    return data[:,FRAME]

def time(data):
    return data[:,TIME]

def percept_x(data):
    return data[:,PERCEPT_POS_X]

def percept_y(data):
    return data[:,PERCEPT_POS_Y]

def absolute_error(data):
    real_x = data[:,REAL_POS_X]
    real_y = data[:,REAL_POS_Y]
    percept_x = data[:,PERCEPT_POS_X]
    percept_y = data[:,PERCEPT_POS_Y]
    return np.sqrt((real_x - percept_x) ** 2 + (real_y - percept_y) ** 2)

def relative_error(data):
    real_x = data[:,REAL_POS_X]
    real_y = data[:,REAL_POS_Y]
    percept_x = data[:,PERCEPT_POS_X]
    percept_y = data[:,PERCEPT_POS_Y]

    return np.abs(1 - np.sqrt(percept_x ** 2 + percept_y ** 2) / np.sqrt(real_x ** 2 + real_y ** 2))

def percent_error(data):
    return 100 * relative_error(data)

def percept_range(data):
    percept_x = data[:,PERCEPT_POS_X]
    percept_y = data[:,PERCEPT_POS_Y]
    
    miro_x = data[:,MIRO_POSE_X]
    miro_y = data[:,MIRO_POSE_Y]

    return np.sqrt((percept_x - miro_x) ** 2 + (percept_y - miro_y) ** 2)

def real_range(data):
    real_x = data[:,REAL_POS_X]
    real_y = data[:,REAL_POS_Y]
    
    miro_x = data[:,MIRO_POSE_X]
    miro_y = data[:,MIRO_POSE_Y]

    return np.sqrt((real_x - miro_x) ** 2 + (real_y - miro_y) ** 2)

def range_error(data):
    return real_range(data) - percept_range(data)

def absolute_range_error(data):
    return np.abs(real_range(data) - percept_range(data))

def relative_range_error(data):
    return np.abs(1 - percept_range(data) / real_range(data))

def percent_range_error(data):
    return 100 * relative_range_error(data)

def absolute_direction_error(data):
    miro_x = data[:,MIRO_POSE_X]
    miro_y = data[:,MIRO_POSE_Y]
    real_x = data[:,REAL_POS_X]
    real_y = data[:,REAL_POS_Y]
    percept_x = data[:,PERCEPT_POS_X]
    percept_y = data[:,PERCEPT_POS_Y]

    real_direction = np.arctan2(real_y - miro_y, real_x - miro_x)
    percept_direction = np.arctan2(percept_y - miro_y, percept_x - miro_x)

    return np.abs(real_direction - percept_direction)

def percent_direction_error(data):
    return 100 * absolute_direction_error(data) / np.pi

def percept_range_camera(data):
    percept_x = data[:,PERCEPT_POS_X]
    percept_y = data[:,PERCEPT_POS_Y]
    z = np.repeat([BALL_R], data.shape[0])

    camera_x = data[:,CAMERA_POS_X]
    camera_y = data[:,CAMERA_POS_Y]
    camera_z = data[:,CAMERA_POS_Z]

    return np.sqrt((percept_x - camera_x) ** 2 + (percept_y - camera_y) ** 2 + (z - camera_z) ** 2)

def real_range_camera(data):
    real_x = data[:,REAL_POS_X]
    real_y = data[:,REAL_POS_Y]
    z = np.repeat([BALL_R], data.shape[0])

    camera_x = data[:,CAMERA_POS_X]
    camera_y = data[:,CAMERA_POS_Y]
    camera_z = data[:,CAMERA_POS_Z]

    return np.sqrt((real_x - camera_x) ** 2 + (real_y - camera_y) ** 2 + (z - camera_z) ** 2)

def range_error_camera(data):
    return real_range_camera(data) - percept_range_camera(data)

def image_x(data):
    return data[:,IMAGE_POS_X]

def image_y(data):
    return data[:,IMAGE_POS_Y]

def radius(data):
    return data[:,RADIUS]

def percept_azimuth(data):
    percept_x = data[:,PERCEPT_POS_X]
    percept_y = data[:,PERCEPT_POS_Y]

    camera_x = data[:,CAMERA_POS_X]
    camera_y = data[:,CAMERA_POS_Y]

    percept_vec_x = percept_x - camera_x
    percept_vec_y = percept_y - camera_y

    return np.arctan2(percept_vec_y, percept_vec_x)

def real_azimuth(data):
    real_x = data[:,REAL_POS_X]
    real_y = data[:,REAL_POS_Y]

    camera_x = data[:,CAMERA_POS_X]
    camera_y = data[:,CAMERA_POS_Y]

    real_vec_x = real_x - camera_x
    real_vec_y = real_y - camera_y

    return np.arctan2(real_vec_y, real_vec_x)

def azimuth_error(data):
    return real_azimuth(data) - percept_azimuth(data)

def percept_elevation(data):
    percept_x = data[:,PERCEPT_POS_X]
    percept_y = data[:,PERCEPT_POS_Y]
    z = np.repeat([BALL_R], data.shape[0])

    camera_x = data[:,CAMERA_POS_X]
    camera_y = data[:,CAMERA_POS_Y]
    camera_z = data[:,CAMERA_POS_Z]

    percept_vec_x = percept_x - camera_x
    percept_vec_y = percept_y - camera_y
    percept_vec_z = z - camera_z

    return np.arctan2(percept_vec_z, np.sqrt(percept_vec_x ** 2 + percept_vec_y ** 2))


def real_elevation(data):
    real_x = data[:,REAL_POS_X]
    real_y = data[:,REAL_POS_Y]
    z = np.repeat([BALL_R], data.shape[0])

    camera_x = data[:,CAMERA_POS_X]
    camera_y = data[:,CAMERA_POS_Y]
    camera_z = data[:,CAMERA_POS_Z]

    real_vec_x = real_x - camera_x
    real_vec_y = real_y - camera_y
    real_vec_z = z - camera_z

    return np.arctan2(real_vec_z, np.sqrt(real_vec_x ** 2 + real_vec_y ** 2))

def elevation_error(data):
    return real_elevation(data) - percept_elevation(data)

def percept_velocity_direction(data):
    percept_x = data[:,PERCEPT_VEL_X]
    percept_y = data[:,PERCEPT_VEL_Y]
    return np.arctan2(percept_y, percept_x)

def real_velocity_direction(data):
    real_x = data[:,REAL_VEL_X]
    real_y = data[:,REAL_VEL_Y]
    return np.arctan2(real_y, real_x)

def velocity_direction_error(data):
    return real_velocity_direction(data) - percept_velocity_direction(data)

def percept_velocity_length(data):
    percept_x = data[:,PERCEPT_VEL_X]
    percept_y = data[:,PERCEPT_VEL_Y]
    return np.sqrt(percept_x ** 2 + percept_y ** 2)

def real_velocity_length(data):
    real_x = data[:,REAL_VEL_X]
    real_y = data[:,REAL_VEL_Y]
    return np.sqrt(real_x ** 2 + real_y ** 2)

def velocity_length_error(data):
    return real_velocity_length(data) - percept_velocity_length(data)

def absolute_velocity_error(data):
    real_vel_x = data[:,REAL_VEL_X]
    real_vel_y = data[:,REAL_VEL_Y]
    percept_vel_x = data[:,PERCEPT_VEL_X]
    percept_vel_y = data[:,PERCEPT_VEL_Y]
    return np.sqrt((real_vel_x - percept_vel_x) ** 2 + (real_vel_y - percept_vel_y) ** 2)

def relative_velocity_error(data):
    real_vel_x = data[:,REAL_VEL_X]
    real_vel_y = data[:,REAL_VEL_Y]
    percept_vel_x = data[:,PERCEPT_VEL_X]
    percept_vel_y = data[:,PERCEPT_VEL_Y]

    return np.abs(1 - np.sqrt(percept_vel_x ** 2 + percept_vel_y ** 2) / np.sqrt(real_vel_x ** 2 + real_vel_y ** 2))

def percent_velocity_error(data):
    return 100 * relative_velocity_error(data)


def head_lift(data):
    return data[:,HEAD_LIFT]

def ball_relative_direction(data):
    miro_pos_x = data[:,MIRO_POSE_X]
    miro_pos_y = data[:,MIRO_POSE_Y]
    miro_theta = data[:,MIRO_POSE_THETA]
    real_x = data[:,REAL_POS_X]
    real_y = data[:,REAL_POS_Y]

    to_ball = np.arctan2(real_y - miro_pos_y, real_x - miro_pos_x)

    return to_ball - miro_theta


# index for data functions and axis labels
variables = {
    'frame': (frame, 'frame number (iterations)'),
    'time': (time, 'time (s)'),
    'percept_x': (percept_x, 'percept x (m)'),
    'percept_y': (percept_y, 'percept y (m)'),
    'absolute_error': (absolute_error, 'absolute error (m)'),
    'relative_error': (relative_error, 'relative error'),
    'percent_error': (percent_error, 'percent error (%)'),
    'percept_range': (percept_range, 'percept range (m)'),
    'real_range': (real_range, 'real range (m)'),
    'range_error': (range_error, 'range error (m)'),
    'absolute_range_error': (absolute_range_error, 'absolute range error (m)'),
    'relative_range_error': (relative_range_error, 'relative range error'),
    'percent_range_error': (percent_range_error, 'percent range error (%)'),
    'absolute_direction_error': (absolute_direction_error, 'absolute direction error (rad)'),
    'percent_direction_error': (percent_direction_error, 'percent direction error (%)'),
    'percept_range_camera': (percept_range_camera, 'percept range (from camera) (m)'),
    'real_range_camera': (real_range_camera, 'real range (from camera) (m)'),
    'range_camera_error': (range_error_camera, 'range error (from camera) (m)'),
    'image_x': (image_x, 'image x (px)'),
    'image_y': (image_y, 'image y (px)'),
    'radius': (radius, 'radius (px)'),
    'percept_azimuth': (percept_azimuth, 'percept azimuth (rad)'),
    'real_azimuth': (real_azimuth, 'real azimuth (rad)'),
    'azimuth_error': (azimuth_error, 'azimuth error (rad)'),
    'percept_elevation': (percept_elevation, 'percept elevation (rad)'),
    'real_elevation': (real_elevation, 'real elevation (rad)'),
    'elevation_error': (elevation_error, 'elevation error (rad)'),
    'percept_velocity_direction': (percept_velocity_direction, 'percept velocity direction (rad)'),
    'real_velocity_direction': (real_velocity_direction, 'real velocity direction (rad)'),
    'velocity_direction_error': (velocity_direction_error, 'velocity direction error (rad)'),
    'percept_velocity_length': (percept_velocity_length, 'percept velocity length (m/s)'),
    'real_velocity_length': (real_velocity_length, 'real velocity length (m/s)'),
    'velocity_length_error': (velocity_length_error, 'velocity length error (m/s)'),
    'head_lift': (head_lift, 'head lift (rad)'),
    'ball_relative_direction': (ball_relative_direction, 'ball relative direction (rad)'),
    'absolute_velocity_error': (absolute_velocity_error, 'absolute velocity error (m/s)'),
    'relative_velocity_error': (relative_velocity_error, 'relative velocity error'),
    'percent_velocity_error': (percent_velocity_error, 'percent velocity error (%)'),
}


def mean(data):
    data = data[~np.isnan(data)]
    return np.mean(data)

def load_data(index):
    """
    Return the observation data, or None if it can't be found
    """
    try:
        data = load(DATA_PATH)
        if index == 'data':
            for i, d in enumerate(data):
                print(i, d.shape)
            return None
        try:
            return data[int(index)]
        except (IndexError, ValueError):
            print('invalid index: %s' % str(index))
            return None
    except FileNotFoundError:
        print('could not find %s' % DATA_PATH)
        return None


if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[1] in ['h', 'help', 'h']:
        print(help)
    else:
        data = load_data(sys.argv[1])
        if data is not None:
            # data to be plotted
            vars = []
            # axis labels
            labels = []
            # filter to select only the desired data
            filter = np.repeat(True, data.shape[0])
            # groups the data to be drawn in different colours
            groups = []
            # names for the groups
            legend = []
            # whether to use a scatter graph or plot a line
            scatter = True
            # colours for the groups
            colours = ['blue', 'red', 'green']

            if '-exp' in sys.argv[2:]:
                index = sys.argv.index('-exp')
                sys.argv = sys.argv[:index] + \
                    ['-e:1', '-#red', '-e:2', '-#green', '-e:3', '-#blue', '-e:4', '-#pink', '-e:5', '-#lime', '-e:6', '-#cyan', '-plot'] \
                    + sys.argv[index+1:]

            for arg in sys.argv[2:]:
                if arg[0] == '-':
                    # choose which groups to display
                    if len(arg) >= 2:
                        group_end = len(arg)
                        tag = 0
                        tag_name = ''
                        tag_select = np.repeat(True, data.shape[0])
                        if ':' in arg:
                            group_end = arg.find(':')
                            tag = int(arg[group_end + 1:])
                            tag_name = ':%d' % tag
                            tag_select = data[:,TAG] == tag
                        if arg[1:group_end] in ['left', 'l']:
                            groups.append(np.bitwise_and(tag_select, data[:,STREAM_INDEX] == 0))
                            # groups.append(data[:,STREAM_INDEX] == 0)
                            legend.append('left%s' % tag_name)
                        elif arg[1:group_end] in ['right', 'r']:
                            groups.append(np.bitwise_and(tag_select, data[:,STREAM_INDEX] == 1))
                            legend.append('right%s' % tag_name)
                        elif arg[1:group_end] in ['observation', 'observe', 'o']:
                            groups.append(np.bitwise_and(tag_select, np.bitwise_or(data[:,STREAM_INDEX] == 0, data[:,STREAM_INDEX] == 1)))
                            legend.append('perception%s' % tag_name)
                        elif arg[1:group_end] in ['estimation', 'estimate', 'e']:
                            groups.append(np.bitwise_and(tag_select, data[:,STREAM_INDEX] == 2))
                            legend.append('estimation%s' % tag_name)
                    # choose a colour for the latest group
                    if len(arg) >= 3:
                        if arg[1] == '#':
                            if len(colours) < len(groups):
                                colours.append(0)
                            colours[len(groups)-1] = arg[2:]
                    # choose options for the latest variable
                    if len(arg) >= 4:
                        if arg[1:3] == 'lt':
                            n = float(arg[3:])
                            filter = np.bitwise_and(filter, vars[-1] < n)
                        elif arg[1:3] == 'gt':
                            n = float(arg[3:])
                            filter = np.bitwise_and(filter, vars[-1] > n)
                        elif arg[1:] == 'deg' or arg[1:] == 'degrees':
                            vars[-1] = np.degrees(vars[-1])
                            labels[-1] = labels[-1].replace('(rad)', '(deg)')
                        elif arg[1:] == 'abs':
                            vars[-1] = np.abs(vars[-1])
                        # choose whether to draw scatter or plot
                        elif arg[1:] == 'plot':
                            scatter = False
                        elif arg[1:] == 'scatter':
                            scatter = True
                else:
                    # choose the variable to be plotted
                    if arg in variables:
                        var, lab = variables[arg]
                        vars.append(var(data))
                        labels.append(lab)
                    else:
                        print('invalid parameter %s' % arg)

            if 2 <= len(vars) <= 3:
                fig = plt.figure(figsize=(16, 8))
                if len(vars) == 3:
                    ax = Axes3D(fig)

                # if no groups were chosen just plot all data
                if len(groups) == 0:
                    groups.append(np.repeat(True, data.shape[0]))
                    legend.append('all data')

                for i, group in enumerate(groups):
                    # selector for the desired data points
                    condition = np.bitwise_and(group, filter)

                    x = vars[0][condition]
                    y = vars[1][condition]
                    legend[i] += ' (avg: %.3f)' % mean(y)
                    # 3d plot
                    if len(vars) == 3:
                        ax.set_xlabel(labels[0])
                        ax.set_ylabel(labels[1])
                        ax.set_zlabel(labels[2])
                        z = vars[2][condition]
                        if scatter:
                            ax.scatter3D(x, y, z, color=colours[i%len(colours)], label=legend[i])
                        else:
                            ax.plot3D(x, y, z, color=colours[i%len(colours)], label=legend[i])
                    # 2d plot
                    else:
                        plt.xlabel(labels[0])
                        plt.ylabel(labels[1])
                        if scatter:
                            plt.scatter(x, y, color=colours[i%len(colours)], label=legend[i])
                        else:
                            plt.plot(x, y, color=colours[i%len(colours)], label=legend[i])

                    print('average %s (%s): %.3f' % (labels[1], legend[i], mean(y)))

                plt.legend(loc='upper right')
                plt.show()
            else:
                print('not 2 or 3 variables!')
                    