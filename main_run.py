# coding=utf-8
import time
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import CompressedImage, PointCloud2, Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from rec_hsm import load_camera_param, HSM_loading, elaboration_param, mask_disp
from rec_hsm import rectification_resize, HSM_computation, point_cloud, xyzrgb_array_to_pointcloud2, point2D
from rec_hsm import generate_oriented_cylinder, distance_calculation, Transform_came2ecm, add_gauge_logo
import os
from cv_bridge import CvBridge
from sksurgerytorch.models import high_res_stereo
import matplotlib.pyplot as plt
class image_feature:

    def __init__(self):       
        self.bridge = CvBridge()   # Image msg to Opencv image conversion (not usable for CompressedImage msg)
        # Camera parameters
        self.Kl = None
        self.Kr = None
        self.dist_l = None
        self.dist_r = None
        self.R = None
        self.T = None
        # Hand-eye calibration matrix (ecm to left camera)
        self.rvec = np.array( [-0.12878932, -0.05637613, 3.11847281])  
        self.tvec = np.array([0.00198865,0.01077903, 0.00980678])    
        
        # Hand-Hand calibration matrix (PSM1 to PSM2)
        
        self.R_PSM1ToPSM2 = np.array([ [0.9988684, 0.01961236, -0.04332785],
                                    [-0.02036795, 0.99964684, -0.01706684],
                                    [ 0.04297783, 0.01792998, 0.99891514]])   
        self.T_PSM1ToPSM2 = np.array([[-0.04165799],
                                    [-0.0270688 ],
                                    [-0.01700285]])  
          
        # Tools position in ECM's RF
        self.tool1 = None
        self.rcm1 = None
        self.tool2 = None
        self.rcm2 = None
        # Matrix computed from camera parameters
        # R_l: performs a change of basis from the unrectified first camera's coordinate system to the rectified first camera's coordinate system
        self.R_l = None
        self.Q = None
        # Parameters for rectification
        self.mapx_l = None
        self.mapy_l = None
        self.mapx_r = None
        self.mapy_r = None

        self.model_disparity = None
        self.model_segmentation = None

        # Image RESOLUTION (original and new)
        self.w = 1920
        self.h = 1080
        self.w_res =  640 #640 #1280 #640  
        self.h_res =  360 #360 #1024 #480 
        # Frame dictionaries
        self.dictL = {}
        self.dictR = {}
        # Last 2 frames keys
        self.secondKL = None
        self.firstKL = None
        self.secondKR = None
        self.firstKR = None
        #### 
        self.img_vis=None

        
    # Callback LEFT image
    def callback_left(self, ros_data):
        # Conversion from ROS CompressedImage msg to Opencv image
        # Take the key
        kl = int(ros_data.header.stamp.to_nsec() // 10e6)       # CENTESIMI sec from epoch
        # Take the image (conversion completed in computation cause it takes more than 10 ms)
        #self.dictL[kl] = np.fromstring(ros_data.data, np.uint8)
        self.dictL[kl]=np.frombuffer(ros_data.data, np.uint8)
        # Save keys of the last 2 frames
        self.secondKL = self.firstKL
        self.firstKL = kl     
         
    def display_image(self):
           # Visualizza l'immagine con matplotlib
       plt.imshow(cv2.cvtColor(self.img_vis, cv2.COLOR_BGR2RGB))
       plt.axis('off')  # Nascondi gli assi
       plt.show()
    # Callback RIGHT image
    def callback_right(self, ros_data):
        # Conversion from ROS Compressed Image msg to Opencv image
        # Take the key
        kr = int(ros_data.header.stamp.to_nsec() // 10e6)
        # Take the image (conversion completed in computation cause it takes more than 10 ms)
        #self.dictR[kr] = np.fromstring(ros_data.data, np.uint8)
        self.dictR[kr]=np.frombuffer(ros_data.data, np.uint8)

        # Save keys of the last 2 frames
        self.secondKR = self.firstKR
        self.firstKR = kr

    # callback to get positions of the tools to backproject on the image
    def callback_PSM1(self, ros_data):
        pos = ros_data.pose.position
        self.tool1 = np.array([pos.x, pos.y, pos.z])
        

    def callback_PSM2(self, ros_data):
        pos = ros_data.pose.position
        self.tool2 = np.array([pos.x, pos.y, pos.z])

    def callback_rcm2(self, ros_data):
        pos = ros_data.pose.position
        self.rcm2 = np.array([pos.x, pos.y, pos.z])

    def callback_rcm1(self, ros_data):
        pos = ros_data.pose.position
        self.rcm1 = np.array([pos.x, pos.y, pos.z])

    def computation(self): 
        # ---------- Preprocessing ----------        
        
        # Clean dict every 25 frames (to free memory) but keep last 2 frames sx and dx
        if len(self.dictL.keys()) > 100 :
            L = [self.dictL[self.secondKL],self.dictL[self.firstKL]]
            R = [self.dictR[self.secondKR],self.dictR[self.firstKR]]
            self.dictL.clear()
            self.dictR.clear()
            self.dictL[self.secondKL], self.dictL[self.firstKL] = L
            self.dictR[self.secondKR], self.dictR[self.firstKR] = R
        
        # Take last frame from dictionaries and relative pose
        # Let's complete the conversion from ros msg to Opencv image
        original_L = cv2.imdecode(self.dictL[self.firstKL], cv2.IMREAD_COLOR)
        print(original_L.shape)
        original_R = cv2.imdecode(self.dictR[self.firstKR], cv2.IMREAD_COLOR)
        
        # original_L = cv2.resize(original_L, (self.w_res,self.h_res), interpolation = cv2.INTER_CUBIC)
        # original_R = cv2.resize(original_R, (self.w_res,self.h_res), interpolation = cv2.INTER_CUBIC)

        #---------------------------------------------------------
        # imgL1 = cv2.cvtColor(original_L, cv2.COLOR_RGB2BGR)
        # imgR1 = cv2.cvtColor(original_R, cv2.COLOR_RGB2BGR)
        imgL1 = original_L
        imgR1 = original_R
        
        
        #--------------------- Align PSM1 to PSM2 -----------------
        #print('self.tool2',self.tool2)
        #print('self.tool1',self.tool1)
        tool1_aligned =  np.dot(self.R_PSM1ToPSM2,  self.tool1.reshape(3,1)) + self.T_PSM1ToPSM2
        rcm1_aligned  = np.dot(self.R_PSM1ToPSM2,  self.rcm1.reshape(3,1)) + self.T_PSM1ToPSM2
        #print("np.dot(self.R_PSM1ToPSM2,  self.tool1)=",np.dot(self.R_PSM1ToPSM2,  self.tool1).shape)
        #print("self.T_PSM1ToPSM2=",self.T_PSM1ToPSM2.shape)
        #print('tool1_aligned',tool1_aligned)
        
        # -------------------- check marker ---------------------
        # p_tip2 = point2D(self.tool2, self.rvec, self.tvec, self.Kl, self.dist_l)
        # p_rcm2 = point2D(self.rcm2, self.rvec, self.tvec, self.Kl, self.dist_l)
        # p_rcm2 = p_tip2 - 600*np.subtract(p_rcm2,p_tip2)/np.linalg.norm(np.subtract(p_rcm2,p_tip2))
        # rcm_2 = np.array([int(p_rcm2[0]),int(p_rcm2[1])])
        # cv2.drawMarker(original_L, tuple(rcm_2) ,color=(0,255,255), markerType=cv2.MARKER_STAR, thickness=5)
        # cv2.drawMarker(original_L, tuple(p_tip2),color=(0,0,255), markerType=cv2.MARKER_CROSS, thickness=5)
        
        # p_tip1 = point2D(tool1_aligned, self.rvec, self.tvec, self.Kl, self.dist_l)
        # p_rcm1 = point2D(rcm1_aligned, self.rvec, self.tvec, self.Kl, self.dist_l)
        # p_rcm1 = p_tip1 - 600*np.subtract(p_rcm1,p_tip1)/np.linalg.norm(np.subtract(p_rcm1,p_tip1))
        # rcm_1 = np.array([int(p_rcm1[0]),int(p_rcm1[1])])
        # cv2.drawMarker(original_L, tuple(rcm_1) ,color=(0,255,255), markerType=cv2.MARKER_STAR, thickness=5)
        # cv2.drawMarker(original_L, tuple(p_tip1),color=(0,0,255), markerType=cv2.MARKER_CROSS, thickness=5)

        # # Display the image using matplotlib
        # import matplotlib.pyplot as plt
        # plt.imshow(original_L)
        # plt.axis('off')  # Hide axes for cleaner display
        # plt.show()
        
        # cv2.imshow('frame_marker',original_L) 
        # cv2.waitKey(10)
        # cv2.destroyAllWindows
        
        # ---------- Rectification and resizing ----------
        imgL_resized = rectification_resize(imgL1, self.mapx_l, self.mapy_l, self.h_res, self.w_res)
        imgR_resized = rectification_resize(imgR1, self.mapx_r, self.mapy_r, self.h_res, self.w_res)

        # ---------- Disparity computation ----------
        #disparity_hsm = HSM_computation(self.model_disparity, imgL_resized, imgR_resized, self.h_res, self.w_res)
        disparity_hsm,_= self.model.predict(imgL_resized, imgR_resized)
        #mport matplotlib.pyplot as plt
        # plt.imshow(disparity_hsm)
        # plt.axis('off')  # Hide axes for cleaner display
        # plt.show()
        #----------- Mask test ------------
        seg_mask = np.zeros_like(disparity_hsm)
        seg_mask2 = imgL_resized
        seg_mask[int(disparity_hsm.shape[0]*0.2):int(disparity_hsm.shape[0]*0.4), int(disparity_hsm.shape[1]*0.5):int(disparity_hsm.shape[1]*0.7)] = 255
        seg_mask2[int(disparity_hsm.shape[0]*0.2):int(disparity_hsm.shape[0]*0.4), int(disparity_hsm.shape[1]*0.5):int(disparity_hsm.shape[1]*0.7)] = 255
        # plt.imshow(seg_mask2)
        # plt.axis('off')  # Hide axes for cleaner display
        # plt.show()
        disparity_hsm = mask_disp(disparity_hsm, seg_mask)

       
        #cv2.imshow('disparity_hsm.png',disparity_hsm)
        #cv2.waitKey(10)
        # ---------- Point cloud computation ----------
        points_3D_hsm = point_cloud(disparity_hsm, self.Q, self.h_res, self.w_res)

        # Remove invalid 3D points (where depth == 0)       
        valid_depth_ind = np.where(points_3D_hsm[:,2].flatten() > 0)[0]
        points_3D_hsm = points_3D_hsm[valid_depth_ind,:]
        
        colors = cv2.cvtColor(imgL_resized, cv2.COLOR_BGR2RGB)
        colors = np.reshape(colors,(self.h_res*self.w_res,3))
        colors = colors[valid_depth_ind,:]

        # ------- Point cloud of surgical scene transfers from rectified camera system to ECM
        points_3D_hsm = np.dot(np.linalg.inv(self.R_l), points_3D_hsm.T)
        T_came2ecm = Transform_came2ecm(self.rvec, self.tvec)
        points_3D_hsm = (np.dot(T_came2ecm[:3,:3], points_3D_hsm) + T_came2ecm[:3,3].reshape(3,1)).T
        
        # ------- Generate cylinder for instruments --------
        PSM2_middle_line, cylinder_PSM2 = generate_oriented_cylinder(self.tool2, self.rcm2, line_length = 0.005, circle_points = 20, radius = 0.004)
        PSM1_middle_line, cylinder_PSM1 = generate_oriented_cylinder(self.tool1, self.rcm1, line_length = 0.005, circle_points = 20, radius = 0.004)        
        
        # ------- Distance Calculation ------------
        distance_value_2, points_extra_2, points_color_extra_2, scene_min_dis_ind_2 = distance_calculation(PSM2_middle_line, cylinder_PSM2, points_3D_hsm, k=1, distance_line_num = 200)
        distance_value_1, points_extra_1, points_color_extra_1, scene_min_dis_ind_1 = distance_calculation(PSM1_middle_line, cylinder_PSM1, points_3D_hsm, k=1, distance_line_num = 200)        
        
        print("The minimum distance of left tool: %f and right tool: %f" % (distance_value_2, distance_value_1))
        
        # ------- Point cloud visualization ----------
        # Rviz visualization
        colors[scene_min_dis_ind_2, :] = [255,0,0]
        colors[scene_min_dis_ind_1, :] = [255,0,0]
        points_pub = np.vstack((points_3D_hsm, points_extra_2, points_extra_1))
        points_color_pub = np.vstack((colors, points_color_extra_2, points_color_extra_1))
        cloud_msg = xyzrgb_array_to_pointcloud2(points_pub, points_color_pub, 'map', stamp=None, seq=None)
        pub_point.publish(cloud_msg)

        #-------- Visualize the image with logo ----------
        self.img_vis = add_gauge_logo(original_L, distance_value_2, distance_value_1)
        cv2.imwrite("a.png", self.img_vis)
        #self.display_image()
        #pub_image.publish(self.bridge.cv2_to_imgmsg(self.img_vis, "bgr8"))



        #pub_image.publish(img_ros)

        cv2.imshow("im.png",self.img_vis)
        cv2.waitKey(10)
        cv2.destroyAllWindows
        
        return 
        

    
if __name__ == '__main__':
    rospy.init_node('distance_vis', anonymous=True)
    rate = rospy.Rate(20)    

    print('\nStarting initialization ...')
    # Class initialization
    ic = image_feature()

    loadweight_disparity = './final-768px.tar' 
    max_disp = 192

    # Subscribers to left and right frames topics
    subscriber_left = rospy.Subscriber("/endoscope/raw/left/image_raw/compressed", CompressedImage, ic.callback_left, queue_size = 1)
    subscriber_right = rospy.Subscriber("/endoscope/raw/right/image_raw/compressed", CompressedImage, ic.callback_right, queue_size = 1)
    
    # Subscribers to the position of the PSMs
    subscriber_PSM1 = rospy.Subscriber("/dvrk/PSM1/position_cartesian_current", PoseStamped, ic.callback_PSM1, queue_size = 1)  
    subscriber_PSM2 = rospy.Subscriber("/dvrk/PSM2/position_cartesian_current", PoseStamped, ic.callback_PSM2, queue_size = 1) 
    subscriber_rcm1 = rospy.Subscriber("/dvrk/SUJ/PSM1/position_cartesian_current", PoseStamped, ic.callback_rcm1, queue_size = 1)
    subscriber_rcm2 = rospy.Subscriber("/dvrk/SUJ/PSM2/position_cartesian_current", PoseStamped, ic.callback_rcm2, queue_size = 1)

    # Publisher to point cloud topic
    pub_point = rospy.Publisher('point_cloud', PointCloud2, queue_size=1)
    pub_image = rospy.Publisher('image_with_target_highlight', Image, queue_size=1)
    #------ PRELIMINARY OPERATIONS --------
    print('Loading model ...')
    # Load HSM model
    #ic.model_disparity = HSM_loading(loadweight_disparity, max_disp)
    max_disp = 255
    entropy_threshold = -1
    scale_factor = 1.0
    level = 1

    ic.model = high_res_stereo.HSMNet(max_disp=max_disp,
                                    entropy_threshold=entropy_threshold,
                                    level=level,
                                    scale_factor=scale_factor,
                                    weights=loadweight_disparity)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Load camera parameters
    [ic.Kl, ic.Kr, ic.dist_l, ic.dist_r, ic.R, ic.T] = load_camera_param('./CalibrationParameters.xml')
    
    # Elaborate camera parameters (compute Q for depth computation and maps for rectification)
    [ic.Q, ic.mapx_l, ic.mapy_l, ic.mapx_r, ic.mapy_r, ic.R_l] = elaboration_param(ic.Kl, ic.Kr, ic.dist_l, ic.dist_r, ic.R, ic.T,
                                                                           ic.h, ic.w, ic.h_res, ic.w_res)

    #--------- 3D RECONSTRUCTION FOR EACH FRAME ---------
    print('Starting reconstructions ...')

    while not rospy.is_shutdown() : 
        
        if len(ic.dictL.keys()) > 1 and ic.tool2 is not None :
            start_time = time.time()
            ic.computation()           
            print('Time for one frame is: %f' % (time.time()-start_time))



        rate.sleep()
 









