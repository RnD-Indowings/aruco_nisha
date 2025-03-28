import rclpy
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import TransformBroadcaster, TransformException
from geometry_msgs.msg import TransformStamped
from mybot_msgs.srv import GetTransform  # Corrected import
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf_transformations import quaternion_from_euler, quaternion_multiply,  quaternion_inverse

# static transform will publish tf messages between base and top
# dynamic transform will publish tf messages every 0.1 seconds between odom and base

class SimpleTfKinematics(Node):
    def __init__(self):
        # base part
        super().__init__("simple_tf_kinematics")
        
        # To publish transform between two reference fixed frames in which the relative position and orientation don't change
        self.static_tf_broadcaster_ = StaticTransformBroadcaster(self)  # we can use this to publish the transform static tf
        self.dynamic_tf_broadcaster_ = TransformBroadcaster(self)  # dynamic tf
        
        # TF static topic is called TransformStamped defined in Geometry msgs
        self.static_transform_stamped_ = TransformStamped()
        # dynamic tf
        self.dynamic_transform_stamped_ = TransformStamped()
        
        # support variables for simulation between odom and mainbase
        # indicates adding transform at any iteration of the timer
        self.x_increment_ = 0.05  # child frame id of dynamic will move 0.05cm
        self.last_x_ = 0.00  # contains the last main value of the x direction of the dynamic that used to calculate the next values
        
        #support variables for rotating while translation *EULER - QUATERNION*
        self.rotations_counter_ = 0
        self.rotations_speed_ = 0.5  # rotating speed in radians per second
        #in terms of roll, pitch and yaw, we can then convert that thing into quaternions  - Library called TF TRANSFORMATIONS
        self.last_orientation_ = quaternion_from_euler(0, 0, 0)   #takes 3 euler angles to convert from EULER TO QUATERNION
        self.orientation_increment_ = quaternion_from_euler(0, 0, 0.05)  #contains the value of the increment in the translation along the x-axis of the robot base frame - this will be added to the transformation whenever the timer expires
        
        
        # creating two more variables for the server
        self.tf_buffer_ = Buffer()
        self.tf_listener_ = TransformListener(self.tf_buffer_, self)
        
        # dynamic transform to publish repeatedly after the get_logger function (line 59)
        
        # adding info about the time in which the transform is generated
        self.static_transform_stamped_.header.stamp = self.get_clock().now().to_msg()
        
        # defining the names of the two frames which are connected by the static transforms
        
        # defining the main base frame id
        self.static_transform_stamped_.header.frame_id = "mybot_base"  # parent for static
        # defining the child frame id
        self.static_transform_stamped_.child_frame_id = "mybot_top"
        
        # Characteristics of this connection to define now
        
        # translation vectors
        self.static_transform_stamped_.transform.translation.x = 0.0
        self.static_transform_stamped_.transform.translation.y = 0.0
        self.static_transform_stamped_.transform.translation.z = 0.3  # making the top frame 30cm away from the base of the mybot which is the MODEL
        
        # representation of the orientations are done using Quaternions
        # Instead of rotational matrices, Euler angles are also used to define the same.
        
        # rotational vectors
        # Quaternions are composed of 4 components and down below we are defining those components (x4)
        # By this - The simple transform between the two frames are done.
        
        #unitary quaternion
        self.static_transform_stamped_.transform.rotation.x = 0.0  # empty rotation setting
        self.static_transform_stamped_.transform.rotation.y = 0.0
        self.static_transform_stamped_.transform.rotation.z = 0.0
        self.static_transform_stamped_.transform.rotation.w = 1.0 
        
        # info adding to the dynamic transform which we know 
        self.dynamic_transform_stamped_.header.stamp = self.get_clock().now().to_msg()
        # setting the frame id *new frame*
        self.dynamic_transform_stamped_.header.frame_id = "odom"  # odometry being new dynamic parent
        # child frame id
        self.dynamic_transform_stamped_.child_frame_id = "mybot_base"
        
        # using static_tf_broadcaster to publish the transform (calling) and sending the static transformed stamped message created above
        self.static_tf_broadcaster_.sendTransform(self.static_transform_stamped_)
        
        # printing a message to inform the user of correct transformation
        self.get_logger().info("Successfully stamped message published between %s and %s" %
                               (self.static_transform_stamped_.header.frame_id, self.static_transform_stamped_.child_frame_id))
        
        # creating a timer for dynamic transform to publish repeatedly
        self.timer_ = self.create_timer(0.1, self.timerCallBack)  # every 0.1 seconds, a new transform is going to be published
        
        # new service server to call the variable get_transform_service and initializing it with create_service_function (takes input the comm interface that is used to comm with the service server)
        self.get_transform_srv_ = self.create_service(GetTransform, "get_transform", self.getTransformCallback)  # function and name of the service server which the other 2 ROS nodes can use, add the callback func too.
        
        # defining the communication interface with the service server - GetTransform
        # in the mybot_msgs folder and the filename is called: GetTransform.srv
        # to get the transform between the frames available in TF and TF Static, importing 2 classes from libraries - TF LISTENER and BUFFER CLASS
        
    def timerCallBack(self):
        self.dynamic_transform_stamped_.header.stamp = self.get_clock().now().to_msg()  # current ROS2 time
        self.dynamic_transform_stamped_.header.frame_id = "odom"  # parent for dynamic
        self.dynamic_transform_stamped_.child_frame_id = "mybot_base"  # child for dynamic
        
        # properties
        self.dynamic_transform_stamped_.transform.translation.x = self.last_x_ + self.x_increment_  #(last value + increment)  #WE WANT TO MAKE THIS ROTATE TOO WHILE TRANSLATING!!!!!!!!!!!!!!!!!!
                                                                                                                            #IN NEED OF SOME SUPPORT VARIABLES.......
        self.dynamic_transform_stamped_.transform.translation.y = 0.0 
        self.dynamic_transform_stamped_.transform.translation.z = 0.0
        
        
               #creating a new quaternion
        #for multiplying the 2 quaternions , importing the multipying quaternions library from tf_transformations
        #basicallt multiplying the last orientation with the current one(incremented one) to get the main orientation respect to the top frame with the bottom frame of the robot here.....
        q = quaternion_multiply (self.last_orientation_, self.orientation_increment_) #new orientation of the robot base frame is calculated in q variable..
        self.dynamic_transform_stamped_.transform.rotation.x = q[0]
        self.dynamic_transform_stamped_.transform.rotation.y = q[1]
        self.dynamic_transform_stamped_.transform.rotation.z = q[2]
        self.dynamic_transform_stamped_.transform.rotation.w = q[3]  #a set of rotation matrix confirmed which is the matrix between robot base and the other frame is now given by q
        #obtained by multiplying the last orientation and the increment in the angle in the orientation of the two frames.......
        
        # meaning both the odom and base will be oriented in the same way
        # for simulation between odom and base, support variables are required within constructor of the simple_tf_kinematics (line 24)
        
        # publishing in the tf topic
        self.dynamic_tf_broadcaster_.sendTransform(self.dynamic_transform_stamped_)
        
        
        # updating the last_x_ value
        self.last_x_ += self.x_increment_  # updating the last x position by adding increment value
        # self.last_x_ = self.dynamic_transform_stamped_.transform.translation.x
        
        #updating the new support variables created for the quaternion
        #1 - Updating the rotation counter
        self.rotations_counter_ += 1  # increasing the rotation counter by 1
        # meaning the rotation counter will be increasing by 1 for each call of the timer callback...
        
        #2 - Updating the incremented orientation
        #self.orientation_increment_ = quaternion_multiply(self.orientation_increment_, self.quaternion_increment_)  # multiplying the last incremented orientation with the incremented one to get the new incremented orientation respect to the top frame with the bottom frame of the robot here...
        # meaning the incremented orientation will be the same as the last incremented orientation multiplied by the incremented rotation angle...
        # for simulation between odom and base, support variables are required within constructor of the simple_tf_kinematics (line 24)
        
        #3 updating the last_orientation_ value
        self.last_orientation_ = q # updating the last orientation by assigning the current rotation to it.
        
        #checking whether the rotation counter is bigger than 100
        if self.rotations_counter_ >=100:   #inversing the quaternion
            self.orientation_increment_ = quaternion_inverse(self.orientation_increment_)
            self.rotations_counter_ = 0
            
            
            #after 100 times, the direction of the rotation changes and as calculated the inverse, it goes -100 inverse and vice-versa..
            
            
            
               
        
    def getTransformCallback(self, req, res):  # takes msg input, the request and the response variable like this
        # executing the function here and info message for the terminal for the server
        self.get_logger().info("Req Transform between %s and %s" %(req.frame_id, req.child_frame_id))
        requested_transform = TransformStamped()
        try:  # to calculate the below transform
            # also we need to know which time we want to know the transform between two frames
            requested_transform = self.tf_buffer_.lookup_transform(req.frame_id, req.child_frame_id, rclpy.time.Time())  # allows us to use tf library to know correct current transform matrix in any 2 frames
            # the above line can throw a transform exception and for that reason, a library should be imported within the ros2.
        except TransformException as e:
            self.get_logger().info("Error occurred while transforming %s and %s" %(req.frame_id, req.child_frame_id))
            res.success = False  # success should be False.
            return res
        
        # assign the required transform to the transform variable of the message of the server.
        res.transform = requested_transform  # contains the output of the lookup transform function
        res.success = True
        return res  # the response message to the service client which requested the message.

# defining and declaring the main function
def main():
    rclpy.init()
    simple_tf_kinematics = SimpleTfKinematics()
    rclpy.spin(simple_tf_kinematics)
    simple_tf_kinematics.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
