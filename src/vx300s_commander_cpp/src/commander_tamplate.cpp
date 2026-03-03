#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>


using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;

class Commander
{
    public:
       Commander(std::shared_ptr<rclcpp::Node> node)
       {
            this->node_ = node;
            arm_ = std::make_shared<MoveGroupInterface>(node_, "arm");
            arm_->setMaxVelocityScalingFactor(1.0);
            arm_->setMaxAccelerationScalingFactor(1.0);
            gripper_ = std::make_shared<MoveGroupInterface>(node, "gripper");

       }



    private:
       void planAndExecute(const std::shared_ptr<MoveGroupInterface> &interface)
       {
            MoveGroupInterface::Plan plan;
            bool success = (interface->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
       }

       
       std::shared_ptr<rclcpp::Node> node_;
       std::shared_ptr<MoveGroupInterface> arm_;
       std::shared_ptr<MoveGroupInterface> gripper_;

};





int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("commander");
    auto commander = Commander(node);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}