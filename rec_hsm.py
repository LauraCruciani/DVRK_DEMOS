# coding=utf-8
import os
import cv2
import copy
import numpy as np 
import math
from math import floor
import time
from model import hsm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model.submodule import *
from utils.preprocess import get_transform
from matplotlib import pyplot as plt
from sensor_msgs.msg import CompressedImage, PointCloud2, PointField
import point_cloud_utils as pcu

#--------------- PRELIMINARY OPERATIONS -------------
#---- Load camera parameters ----
def load_camera_param (xml_file):
    #print(xml_file)
    cv_file = cv2.FileStorage(xml_file, cv2.FILE_STORAGE_READ)
    # The intrinsic parameters (intrinsic matrix), respectively for left and right camera
    Kl = cv_file.getNode("M_l").mat()
    Kr = cv_file.getNode("M_r").mat()
    # The distortion parameters, respectively for left and right camera
    dist_l = cv_file.getNode("D_l").mat().flatten()
    dist_r = cv_file.getNode("D_r").mat().flatten()
    # The extrinsic parameters, respectively rotation and translation matrix
    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()
    cv_file.release()
    return Kl, Kr, dist_l,dist_r, R, T

#---- HSM loading ----
def HSM_loading(loadweight_disparity, max_disp):
    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    #cudnn.benchmark = True
    cudnn.benchmark = False
  
    # construct model
    model = hsm(max_disp, clean=1)
    model = nn.DataParallel(model, device_ids=[0])
    model.cuda()

    pretrained_dict = torch.load(loadweight_disparity)
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)

    model.eval()


    if max_disp % 16 != 0:
        max_disp = 16 * floor(max_disp/16)
    max_disp = int(max_disp)


    ## change max disp

    model.module.maxdisp = max_disp
    if model.module.maxdisp ==64: model.module.maxdisp=128
    model.module.disp_reg8 =  disparityregression(model.module.maxdisp,16).cuda()
    model.module.disp_reg16 = disparityregression(model.module.maxdisp,16).cuda()
    model.module.disp_reg32 = disparityregression(model.module.maxdisp,32).cuda()
    model.module.disp_reg64 = disparityregression(model.module.maxdisp,64).cuda()

    return model

#---- Parameters elaboration ----
def elaboration_param(Kl, Kr, dist_l, dist_r, R, T, h, w, h_res, w_res):
    
    # Computation of matrix Q tha's used for conversion from disparity to depth
    x_ratio = w_res / w
    y_ratio = h_res / h
    [R_l, R_r, P_l, P_r, _] = stereoRectify(Kl, Kr, dist_l, dist_r, h, w, R, T)
    Kl_res = copy.copy(Kl)
    Kr_res = copy.copy(Kr)
    Kl_res[0,:] = Kl_res[0,:] * x_ratio
    Kl_res[1,:] = Kl_res[1,:] * y_ratio
    Kr_res[0,:] = Kr_res[0,:] * x_ratio
    Kr_res[1,:] = Kr_res[1,:] * y_ratio
    [_,_,_,_,Q] = stereoRectify(Kl_res, Kr_res, dist_l, dist_r, h_res, w_res, R, T)  
    
    # Computation of maps used for rectification
    mapx_l, mapy_l = cv2.initUndistortRectifyMap(Kl, dist_l, R_l, P_l, (w,h), cv2.CV_16SC2)    #CV_32F
    mapx_r, mapy_r = cv2.initUndistortRectifyMap(Kr, dist_r, R_r, P_r, (w,h), cv2.CV_16SC2)    #CV_32F

    return Q, mapx_l, mapy_l, mapx_r, mapy_r, R_l



#--------------- RECTIFICATION -------------
def stereoRectify(Kl, Kr, dist_l, dist_r, h, w, R, T):
        (R_l, R_r, P_l, P_r, Q, roi1, roi2) = cv2.stereoRectify(cameraMatrix1 = Kl,cameraMatrix2 = Kr,
                                                                distCoeffs1 = dist_l, distCoeffs2 = dist_r,
                                                                imageSize = (h,w),
                                                                R = R, T = T,
                                                                #flags = cv2.CALIB_USE_INTRINSIC_GUESS,
                                                                flags=cv2.CALIB_ZERO_DISPARITY,
                                                                alpha = 0.5,
                                                                newImageSize = (h,w) )
        '''
        cameraMatrix - the two intrinsic matrix
        R : Rotation matrix between the coordinate systems of the first and the second cameras.
        T : Translation vector between coordinate systems of the cameras.
        R1 : Output 3x3 rectification transform (rotation matrix) for the first camera.
        R2 : Output 3x3 rectification transform (rotation matrix) for the second camera.
        P1 : Output 3x4 projection matrix in the new (rectified) coordinate systems for the first camera.
        P2 : Output 3x4 projection matrix in the new (rectified) coordinate systems for the second camera.
        Q : Output 4 \times 4 disparity-to-depth mapping matrix '''
        return R_l, R_r, P_l, P_r, Q

def rectification_resize(img, mapx, mapy, h_res, w_res):
    
    # undistorting the images witht the calculated undistortion map
    result = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    # RESIZE
    img_res = cv2.resize(result, (w_res,h_res), interpolation = cv2.INTER_CUBIC)
    return img_res


def mask_disp(disp, mask):
    ind = mask == 0 
    disp[ind] = mask[ind] 
    return disp

#--------------- HSM DISPARITY COMPUTATION -------------
def HSM_computation(model, imgL_o, imgR_o, h, w):
    
    processed = get_transform()

    imgL = processed(imgL_o).numpy()
    imgR = processed(imgR_o).numpy() 

    imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
    imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

    ##fast pad
    max_h = int(h // 64 * 64)
    max_w = int(w // 64 * 64)
    if max_h < h: max_h += 64
    if max_w < w: max_w += 64

    top_pad = max_h-h
    left_pad = max_w-w
    imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    
    imgL = Variable(torch.FloatTensor(imgL).cuda())
    imgR = Variable(torch.FloatTensor(imgR).cuda())

    with torch.no_grad():
        pred_disp,_ = model(imgL,imgR)
    pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()
    #print((time.time() - start_time))
    
    top_pad   = max_h-imgL_o.shape[0]
    left_pad  = max_w-imgL_o.shape[1]
    #entropy = entropy[top_pad:,:pred_disp.shape[1]-left_pad].cpu().numpy()
    pred_disp = pred_disp[top_pad:,:pred_disp.shape[1]-left_pad]

    invalid = np.logical_or(pred_disp == np.inf,pred_disp!=pred_disp)
    pred_disp[invalid] = np.inf
    torch.cuda.empty_cache()

    return pred_disp


#--------------- 3D POINT CLOUD COMPUTATION -------------
def point_cloud(dispTo3D, Q, h, w) :
    points_3D = cv2.reprojectImageTo3D(dispTo3D, Q)
    points_3D = np.reshape(points_3D,(h*w,3))
    points_3D[~np.isfinite(points_3D)] = 0
    return points_3D
    

# Conversion to PointCloud2 ros mesg
# https://github.com/spillai/pybot/blob/df6989d90860e88de42a3183d3af6c4c1c06336a/pybot/externals/ros/pointclouds.py
def xyzrgb_array_to_pointcloud2(points, colors, frame_id, stamp=None,  seq=None):

    # Rviz wants colors [0,1]
    colors = colors/255

    #Create a sensor_msgs.PointCloud2 from an array of points.
    msg = PointCloud2()
    assert(points.shape == colors.shape)
    '''
    buf = []
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if seq: 
        msg.header.seq = seq
    '''
    msg.header.frame_id = frame_id
    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        N = len(points)
        xyzrgb = np.array(np.hstack([points, colors]), dtype=np.float32)
        msg.height = 1
        msg.width = N

    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('r', 12, PointField.FLOAT32, 1),
        PointField('g', 16, PointField.FLOAT32, 1),
        PointField('b', 20, PointField.FLOAT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 24
    msg.row_step = msg.point_step * N
    msg.is_dense = True; 
    msg.data = xyzrgb.tostring()

    return msg


def point2D(P3D, rvec, tvec, K, dist):
    # Position of the tool's tip on the image (from kinematics)
    imgP = cv2.projectPoints(P3D, rvec, tvec, K, dist)
    imgP = np.array([int(imgP[0][0][0][0]),int(imgP[0][0][0][1])])
    return imgP 

def generate_oriented_cylinder(EE_grip, EE_rcm, line_length, circle_points, radius):
    # Get the vector between the two points
    vec = EE_rcm - EE_grip
    vec_length = np.linalg.norm(vec)
    line_point_num = int(vec_length // line_length)
    line_remain = vec_length % line_length
    # Normalize the vector
    vec = vec / np.linalg.norm(vec)


    # Get a perpendicular vector to the main axis
    perp = np.cross(vec, [1, 0, 0])
    if np.allclose(perp, 0):
        perp = np.cross(vec, [0, 1, 0])
    
    perp = perp / np.linalg.norm(perp)

    # Get the rotation matrix to align the vector with the z-axis
    rot_matrix = np.zeros((3, 3))
    rot_matrix[:, 0] = perp
    rot_matrix[:, 1] = np.cross(vec, perp)
    rot_matrix[:, 2] = vec

    # Generate a set of points on a circle
    theta = np.linspace(0, 2 * np.pi, circle_points, endpoint=False)
    
    if line_remain != 0:
        middle_3d_positions = np.linspace(EE_grip, EE_rcm - line_remain*vec, line_point_num+1)
        
        circle = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.zeros(circle_points)]) 
        circle = np.dot(circle, rot_matrix.T) 
        
        circle_start = circle + middle_3d_positions[0]
        circle_end = circle + middle_3d_positions[-1]

        
        for i in range(circle_points):
            boundary_line = np.linspace(circle_start[i], circle_end[i], line_point_num+1)
            boundary_line = np.vstack((boundary_line, circle + EE_rcm))
            all_lines = boundary_line if i==0 else np.vstack((all_lines, boundary_line))

    else:
        middle_3d_positions = np.linspace(EE_grip, EE_rcm, line_point_num+1)
        
        circle = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.zeros(circle_points)]) 
        circle = np.dot(circle, rot_matrix.T) 
        
        circle_start = circle + middle_3d_positions[0]
        circle_end = circle + middle_3d_positions[-1]

        
        for i in range(circle_points):
            boundary_line = np.linspace(circle_start[i], circle_end[i], line_point_num+1)
            all_lines = boundary_line if i==0 else np.vstack((all_lines, boundary_line))    
    
    return middle_3d_positions, all_lines

def color_list(distance):
    
    colors = [
        [255,51,51],
        [255,153,51],
        [255,255,51],
        [153,255,51],
        [51,255,255],
        [51,153,255],
        [255,0,0]]
    
    if distance > 0.06:
        row = 6
    elif distance <= 0:
        row = 0
    else:
        for i in range(6):
            if 0.01 * i < distance <= 0.01 * i + 0.01:
                row = i
                break
    color = colors[row]
    return np.array(color)
    # Gray = np.array([192,192,192])
    # Blue = np.array([51,51,225])
    # Green = np.array([51,255,51])
    # Yellow = np.array([255,255,51])
    # Orange = np.array([255,153,51])
    # Red = np.array([255,0,0])
    
    # if distance > 0.05:
    #     color = Gray
    # elif 0.05 <= distance <0.04:
    #     color = Blue
    # elif 0.04 <= distance <0.03:
    #     color = Green
    # elif 0.03 <= distance <0.02:
    #     color = Yellow    
    # elif 0.02 <= distance <0.01:
    #     color = Orange
    # else:
    #     color = Red  
    
def distance_calculation(PSM_middle_line, instruments, scene, k, distance_line_num):
    
    # dists_a_to_b is of shape (pts_a.shape[0], k) and contains the (sorted) distances
    # to the k nearest points in pts_b
    # corrs_a_to_b is of shape (a.shape[0], k) and contains the index into pts_b of the
    # k closest points for each point in pts_a    
     
    scene = np.array(scene, order = 'C')
    dists_a_to_b, corrs_a_to_b = pcu.k_nearest_neighbors(np.float32(instruments), np.float32(scene), k)
    index = np.argmin(dists_a_to_b)

    point_instrument = instruments[index,:]
    point_scene = scene[corrs_a_to_b[index],:]
    distance_value = dists_a_to_b[index]
    distance_points = point_instrument + (1 / distance_line_num) * np.arange(1,distance_line_num).reshape(distance_line_num-1,1)*(point_scene - point_instrument)   
    distance_points_color = color_list(distance_value)
    pcd_points = np.vstack((PSM_middle_line, instruments, distance_points))
    
    pcd_instrument_middle_line = np.tile(np.array([255,153,255]), (PSM_middle_line.shape[0],1)) 
    pcd_colors_instrument = np.tile(np.array([0,0,0]), (instruments.shape[0],1))
    pcd_colors_distance = np.tile(distance_points_color, (distance_points.shape[0],1))
    pcd_colors_instrument[index,:] = [255,0,0]

    pcd_colors = np.vstack((pcd_instrument_middle_line, pcd_colors_instrument, pcd_colors_distance)) 

    
    return distance_value, pcd_points, pcd_colors, corrs_a_to_b[index]

def Transform_came2ecm(RVEC, TVEC):
    Recm, _ = cv2.Rodrigues(RVEC)
    transform_matrix = np.hstack((Recm, TVEC.reshape(3,1)))
    Tecm_cam = np.eye(4)
    Tecm_cam[:3, :] = transform_matrix
    Tcam_ecm = np.linalg.inv(Tecm_cam)
    return Tcam_ecm


def add_gauge_logo(im, value_PSM2, value_PSM1):
    
    colors_list = [
    [255,51,51],
    [255,153,51],
    [255,255,51],
    [153,255,51],
    [51,255,255],
    [51,153,255],
    [204,255,204]]
    
    value_PSM2 = value_PSM2 * (-3000)
    value_PSM1 = value_PSM1 * (-3000)
    # Define the gauge center and radius
    height, width, _ = im.shape
    center_PSM2 = (int(width * 0.09), int(height * 0.15))
    center_PSM1 = (int(width * 0.91), int(height * 0.15))
    radius = 100
    
    # Draw the gauge outline
    start_angle = -180
    end_angle = 0
    color_PSM2 = colors_list[-1][::-1]
    color_PSM1 = [229,204,255]
    thickness = 2
    im = cv2.ellipse(im, center_PSM2, (radius, radius), 0, start_angle, end_angle, color_PSM2, thickness)
    im = cv2.ellipse(im, center_PSM1, (radius, radius), 0, start_angle, end_angle, color_PSM1, thickness)

    target_PSM2 = (int(center_PSM2[0] + radius * math.cos(math.radians(value_PSM2))), 
            int(center_PSM2[1] + radius * math.sin(math.radians(value_PSM2))))

    target_PSM1 = (int(center_PSM1[0] + radius * math.cos(math.radians( -180 - value_PSM1))), 
        int(center_PSM1[1] + radius * math.sin(math.radians( - 180 - value_PSM1))))
   
    # Draw the gauge fill of PSM2
    for i in range(6):
        if value_PSM2 < -180:
            break
        elif -180 + 30*i <= value_PSM2 < -180 + 30*(i+1):
            color_target = colors_list[5-i][::-1]
            im = cv2.line(im, center_PSM2, target_PSM2, color_target, thickness=5)
            start_angle = -180
            end_angle = value_PSM2
            color = colors_list[5-i][::-1]
            mask = np.zeros((height, width, 3), dtype=np.uint8)
            mask = cv2.ellipse(mask, center_PSM2, (radius, radius), 0, start_angle, end_angle, color, -1)
            overlay = cv2.addWeighted(mask, 0.4, im, 1, 0)
            im = overlay           
            break
        elif value_PSM2 >= 0:        
            start_angle = -180
            end_angle = 0
            color = [0,0,255]
            mask = np.zeros((height, width, 3), dtype=np.uint8)
            mask = cv2.ellipse(mask, center_PSM2, (radius, radius), 0, start_angle, end_angle, color, -1)
            overlay = cv2.addWeighted(mask, 0.4, im, 1, 0)
            im = overlay
            break   

    # Draw the gauge fill of PSM1
    for i in range(6):
        if value_PSM1 < -180:
            break
        elif -180 + 30*i <= value_PSM1 < -180 + 30*(i+1):
            color_target = colors_list[5-i][::-1]
            im = cv2.line(im, center_PSM1, target_PSM1, color_target, thickness=5)
            start_angle = -180 - value_PSM1
            end_angle = 0
            color = colors_list[5-i][::-1]
            mask = np.zeros((height, width, 3), dtype=np.uint8)
            mask = cv2.ellipse(mask, center_PSM1, (radius, radius), 0, start_angle, end_angle, color, -1)
            overlay = cv2.addWeighted(mask, 0.4, im, 1, 0)
            im = overlay           
            break
        elif value_PSM1 >= 0:        
            start_angle = -180
            end_angle = 0
            color = [0,0,255]
            mask = np.zeros((height, width, 3), dtype=np.uint8)
            mask = cv2.ellipse(mask, center_PSM1, (radius, radius), 0, start_angle, end_angle, color, -1)
            overlay = cv2.addWeighted(mask, 0.4, im, 1, 0)
            im = overlay
            break   

    text = ['SA', '5', '4', 'RA', '2', '1', '0']
    # Draw the gauge tick marks
    count = 0
    
    for i in [-180,-150, -120, -90, -60, -30, 0]:
        
        angle = i
        start_PSM2 = (int(center_PSM2[0] + (radius + 10) * math.cos(math.radians(angle))), 
                 int(center_PSM2[1] + (radius + 10) * math.sin(math.radians(angle))))
        end_PSM2 = (int(center_PSM2[0] + radius * math.cos(math.radians(angle))), 
               int(center_PSM2[1] + radius * math.sin(math.radians(angle))))
        thickness = 2
        im = cv2.line(im, start_PSM2, end_PSM2, color_PSM2, thickness)
        
        start_PSM1 = (int(center_PSM1[0] + (radius + 10) * math.cos(math.radians(-180 - angle))), 
                 int(center_PSM1[1] + (radius + 10) * math.sin(math.radians(-180 - angle))))
        end_PSM1 = (int(center_PSM1[0] + radius * math.cos(math.radians(-180 - angle))), 
               int(center_PSM1[1] + radius * math.sin(math.radians(-180 - angle))))
        im = cv2.line(im, start_PSM1, end_PSM1, color_PSM1, thickness)        
        
        if i == -180: 
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            (text_width, text_height), _ = cv2.getTextSize('PSM2', font, font_scale, thickness)
            text_origin_PSM2 = (int(center_PSM2[0] - text_width / 2),
                        int(center_PSM2[1]+ text_height / 2 + 20))
            im = cv2.putText(im, 'PSM2', text_origin_PSM2, font, font_scale, color_PSM2, thickness, cv2.LINE_AA)
            (text_width, text_height), _ = cv2.getTextSize('PSM1', font, font_scale, thickness)
            text_origin_PSM1 = (int(center_PSM1[0] - text_width / 2),
                        int(center_PSM1[1]+ text_height / 2 + 20))
            im = cv2.putText(im, 'PSM1', text_origin_PSM1, font, font_scale, color_PSM1, thickness, cv2.LINE_AA)            
            
            
        # Add the value text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(text[count], font, font_scale, thickness)
        text_origin_PSM2 = (int(center_PSM2[0] + (radius + 10 + text_height) * math.cos(math.radians(angle)) - text_width / 2),
                      int(center_PSM2[1] + (radius + 10 + text_height) * math.sin(math.radians(angle)) + text_height / 2))
        im = cv2.putText(im, text[count], text_origin_PSM2, font, font_scale, color_PSM2, thickness, cv2.LINE_AA)
        
        (text_width, text_height), _ = cv2.getTextSize(text[6 - count], font, font_scale, thickness)
        text_origin_PSM1 = (int(center_PSM1[0] + (radius + 10 + text_height) * math.cos(math.radians(-180 - angle)) - text_width / 2),
        int(center_PSM1[1] + (radius + 10 + text_height) * math.sin(math.radians(-180 - angle)) + text_height / 2))
        im = cv2.putText(im, text[count], text_origin_PSM1, font, font_scale, color_PSM1, thickness, cv2.LINE_AA)
        
        count += 1
        
    return im







