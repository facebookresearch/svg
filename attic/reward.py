
class ReacherRew(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dims = torch.LongTensor([7,8,9])

    def forward(self, xu):
        if xu.dim() == 3:
            flat_xu = xu.permute(2,0,1).reshape(self.obs_dim+self.action_dim, -1).t()
            r = self.forward(flat_xu)
            r = r.reshape(xu.size(0), xu.size(1))
            return r

        assert xu.dim() == 2
        n_batch, nxu = xu.size()
        x = xu[:,:self.obs_dim]
        u = xu[:,self.obs_dim:]

        if self.mean_obs is not None:
            x = x*self.std_obs + self.mean_obs

        import ipdb; ipdb.set_trace() # Need to debug more
        ee_pos = x[:,-3:]
        # ee_pos = self.get_ee_pos(x)
        # ee_pos_np = self.get_ee_pos_np(utils.to_np(x))

        goal = x[:,self.goal_dims]
        assert ee_pos.size() == goal.size()
        r = -(ee_pos-goal).pow(2).sum(dim=1)
        r -= 0.01*u.pow(2).sum(dim=1)

        return r

    def get_ee_pos(self, states):
        assert states.dim() == 2
        theta1, theta2, theta3, theta4, theta5, theta6, theta7 = \
            states[:, :1], states[:, 1:2], states[:, 2:3], states[:, 3:4], \
            states[:, 4:5], states[:, 5:6], states[:, 6:]
        rot_axis = torch.cat(
            [torch.cos(theta2) * torch.cos(theta1),
             torch.cos(theta2) * torch.sin(theta1), -torch.sin(theta2)], dim=1)
        rot_perp_axis = torch.cat(
            [-torch.sin(theta1), torch.cos(theta1), torch.zeros_like(theta1)], dim=1)
        cur_end = torch.cat([
            0.1 * torch.cos(theta1) + 0.4 * torch.cos(theta1) * torch.cos(theta2),
            0.1 * torch.sin(theta1) + 0.4 * torch.sin(theta1) * torch.cos(theta2) - 0.188,
            -0.4 * torch.sin(theta2)
        ], dim=1)

        for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
            perp_all_axis = torch.cross(rot_axis, rot_perp_axis)
            x = torch.cos(hinge) * rot_axis
            y = torch.sin(hinge) * torch.sin(roll) * rot_perp_axis
            z = -torch.sin(hinge) * torch.cos(roll) * perp_all_axis
            new_rot_axis = x + y + z
            new_rot_perp_axis = torch.cross(new_rot_axis, rot_axis)

            I = torch.norm(new_rot_perp_axis, dim=1) < 1e-30
            new_rot_perp_axis[I] = rot_perp_axis[I]

            new_rot_perp_axis = new_rot_perp_axis / \
              torch.norm(new_rot_perp_axis, dim=1, keepdim=True)
            rot_axis, rot_perp_axis, cur_end = new_rot_axis, new_rot_perp_axis, \
              cur_end + length * new_rot_axis
        return cur_end


    # https://github.com/WilsonWangTHU/POPLIN/blob/53d50db0befaeb86481f4ebb25590e905b613d2e/dmbrl/config/reacher.py#L123
    def get_ee_pos_np(self, states):
        theta1, theta2, theta3, theta4, theta5, theta6, theta7 = \
            states[:, :1], states[:, 1:2], states[:, 2:3], states[:, 3:4], \
            states[:, 4:5], states[:, 5:6], states[:, 6:]
        rot_axis = np.concatenate(
            [np.cos(theta2) * np.cos(theta1),
             np.cos(theta2) * np.sin(theta1), -np.sin(theta2)], axis=1)
        rot_perp_axis = np.concatenate(
            [-np.sin(theta1), np.cos(theta1), np.zeros(theta1.shape)], axis=1)
        cur_end = np.concatenate([
            0.1 * np.cos(theta1) + 0.4 * np.cos(theta1) * np.cos(theta2),
            0.1 * np.sin(theta1) + 0.4 * np.sin(theta1) * np.cos(theta2) - 0.188,
            -0.4 * np.sin(theta2)
        ], axis=1)

        for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
            perp_all_axis = np.cross(rot_axis, rot_perp_axis)
            x = np.cos(hinge) * rot_axis
            y = np.sin(hinge) * np.sin(roll) * rot_perp_axis
            z = -np.sin(hinge) * np.cos(roll) * perp_all_axis
            new_rot_axis = x + y + z
            new_rot_perp_axis = np.cross(new_rot_axis, rot_axis)
            new_rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30] = \
                rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30]
            new_rot_perp_axis /= np.linalg.norm(new_rot_perp_axis, axis=1, keepdims=True)
            rot_axis, rot_perp_axis, cur_end = new_rot_axis, new_rot_perp_axis, \
              cur_end + length * new_rot_axis
        return cur_end