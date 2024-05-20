import sys
import rclpy
from rclpy.node import Node as node
from rclpy.action import ActionClient
import random

from interfaces.srv import Multi
from interfaces.msg import Univid
from interfaces.action import Uid

class node2(node):

    def __init__(self):
        super().__init__('node2')
        self.subscription = self.create_subscription(Univid,'University_ID',self.listener_callback,10) #섭스크라이버 생성
        self.cli = self.create_client(Multi, 'multiply')                                         #서비스 클라이언트 생성
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Multi.Request()
        self._action_client = ActionClient(self, Uid, 'Calc_Action')                           #액션 클라이언트 설정
        self.declare_parameter('uID', '2022741034')                                              #파라미터 설정

    def listener_callback(self, msg):
        self.get_logger().info('Recieved(UID) "%s"' % msg.univid)
        
    def send_request(self, a, b):
        self.req.x = a
        self.req.y = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        
        return self.future.result()
    
    def send_goal(self):
        user_param = self.get_parameter('uID').get_parameter_value().string_value
        
        new_user_param = rclpy.parameter.Parameter(
            'uID',
            rclpy.Parameter.Type.STRING,
            '2022741034'
        )
        all_new_params = [new_user_param]
        self.set_parameters(all_new_params)
        
        goal_msg = Uid.Goal()
        goal_msg.univid = user_param
        
        self._action_client.wait_for_server()
        
        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected...')
            return
        
        self.get_logger().info('Goal Accepted!')
        
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
    
    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Result: {0}'.format(result.process))
    
    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info('Recieved feedback: [First element is partial sum] {0}'.format(feedback.progress))

def main(args=None):
    rclpy.init(args=args)

    nd2 = node2()
    # 무작위 값 생성
    a = random.randint(1, 10)  # 1부터 10 사이의 무작위 정수
    b = random.randint(1, 10)  # 1부터 10 사이의 무작위 정수
    
    response = nd2.send_request(a, b)
    nd2.get_logger().info(
        'Multiply Result: %d * %d = %d' %
        (a, b, response.res))
    
    nd2.send_goal()
    rclpy.spin(nd2)
    nd2.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
