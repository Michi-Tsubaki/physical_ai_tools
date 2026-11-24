from xml.etree import ElementTree

import numpy as np
import PyKDL as kdl

from physical_ai_server.data_processing.task_space import (
    BASE_LINK, BODY_JOINTS, FEATURE_NAMES, TIP_LINKS, pose_vector, rotation_matrix,
)


def joint_array(values):
    result = kdl.JntArray(len(values))
    for i, value in enumerate(values):
        result[i] = value
    return result


def vector(element, attribute, default):
    value = default if element is None else element.get(attribute, default)
    return kdl.Vector(*map(float, value.split()))


class ArmChain:

    def __init__(self, joints, tip, joint_order, limits):
        segments = []
        link = tip
        while link != BASE_LINK:
            joint = joints[link]
            segments.append(joint)
            link = joint.find('parent').attrib['link']
        self.indices = []
        self.chain = kdl.Chain()
        for joint in reversed(segments):
            origin = joint.find('origin')
            frame = kdl.Frame(kdl.Rotation.RPY(*vector(origin, 'rpy', '0 0 0')),
                              vector(origin, 'xyz', '0 0 0'))
            name = joint.attrib['name']
            if joint.attrib['type'] == 'fixed':
                axis_joint = kdl.Joint(name)
            elif joint.attrib['type'] == 'revolute':
                axis = vector(joint.find('axis'), 'xyz', '1 0 0')
                axis_joint = kdl.Joint(name, frame.p, frame.M * axis, kdl.Joint.RotAxis)
                self.indices.append(joint_order.index(name))
            else:
                raise ValueError(f'Unsupported joint type for {name}')
            link = joint.find('child').attrib['link']
            self.chain.addSegment(kdl.Segment(link, axis_joint, frame))

        lower, upper = limits[self.indices].T
        self.fk = kdl.ChainFkSolverPos_recursive(self.chain)
        self.velocity_solver = kdl.ChainIkSolverVel_pinv(self.chain)
        self.ik = kdl.ChainIkSolverPos_NR_JL(
            self.chain, joint_array(lower), joint_array(upper), self.fk, self.velocity_solver)

    def encode(self, joints):
        frame = kdl.Frame()
        if self.fk.JntToCart(joint_array(joints[self.indices]), frame) < 0:
            raise ValueError('Forward kinematics failed')
        return pose_vector(list(frame.p), np.array([[frame.M[i, j] for j in range(3)]
                                                    for i in range(3)]))

    def solve(self, pose, seed):
        orientation = rotation_matrix(pose[3:])
        target = kdl.Frame(kdl.Rotation(*orientation.ravel()), kdl.Vector(*pose[:3]))
        solution = kdl.JntArray(len(self.indices))
        if self.ik.CartToJnt(joint_array(seed[self.indices]), target, solution) < 0:
            raise ValueError('No joint solution for predicted wrist pose')
        return list(solution)


class NextageKinematics:

    def __init__(self, description, joint_order):
        root = ElementTree.fromstring(description)
        joints = {j.find('child').attrib['link']: j for j in root.findall('joint')}
        by_name = {j.attrib['name']: j for j in joints.values()}
        self.limits = np.array([[float(by_name[name].find('limit').attrib[key])
                                 for key in ('lower', 'upper')] for name in joint_order])
        self.body = [joint_order.index(name) for name in BODY_JOINTS]
        self.arms = [ArmChain(joints, tip, joint_order, self.limits) for tip in TIP_LINKS]
        indices = self.body + [i for arm in self.arms for i in arm.indices]
        if sorted(indices) != list(range(len(joint_order))):
            raise ValueError('Joint order does not match the NEXTAGE task space profile')

    def encode(self, joints):
        joints = np.asarray(joints)
        if joints.shape != (len(self.limits),) or not np.isfinite(joints).all():
            raise ValueError('Incomplete or invalid NEXTAGE joint state')
        poses = [arm.encode(joints) for arm in self.arms]
        return np.concatenate((*poses, joints[self.body])).astype(np.float32)

    def decode(self, action, seed):
        action = np.asarray(action)
        if action.shape != (len(FEATURE_NAMES),) or not np.isfinite(action).all():
            raise ValueError('Invalid task space action')
        seed = np.asarray(seed)
        joints = seed.astype(np.float64, copy=True)
        joints[self.body] = action[18:]
        for i, arm in enumerate(self.arms):
            offset = 9 * i
            joints[arm.indices] = arm.solve(action[offset:offset + 9], seed)
        if not np.all((self.limits[:, 0] - 1e-6 <= joints) & (joints <= self.limits[:, 1] + 1e-6)):
            raise ValueError('Predicted joint positions exceed URDF limits')
        return np.clip(joints, *self.limits.T).tolist()
