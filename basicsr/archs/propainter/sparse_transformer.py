import math
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class SoftSplit(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding,gain=0.02):
        super(SoftSplit, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.t2t = nn.Unfold(kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
        c_in = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(c_in, hidden)
        self.init_weights(gain=gain)
    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1
                                           or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' %
                        init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)
    def forward(self, x, b, output_size):
        f_h = int((output_size[0] + 2 * self.padding[0] -
                   (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        f_w = int((output_size[1] + 2 * self.padding[1] -
                   (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        feat = self.t2t(x)
        # feat shape [b*t, ks*ks*c，num_vec ]
        feat = feat.permute(0, 2, 1)
        # feat shape [b*t, num_vec, ks*ks*c]
        feat = self.embedding(feat)
        # feat shape after embedding [b, t*num_vec, hidden]
        feat = feat.view(b, -1, f_h, f_w, feat.size(2))
        return feat

class SoftComp(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding,gain=0.02):
        super(SoftComp, self).__init__()
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_conv = nn.Conv2d(channel,
                                   channel,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.init_weights(gain=gain)
    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1
                                           or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' %
                        init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)
    def forward(self, x, t, output_size):
        b_, _, _, _, c_ = x.shape
        x = x.view(b_, -1, c_)
        feat = self.embedding(x)
        b, _, c = feat.size()
        feat = feat.view(b * t, -1, c).permute(0, 2, 1)
        feat = F.fold(feat,
                      output_size=output_size,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding)
        feat = self.bias_conv(feat)
        return feat





class FusionFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=1000, t2t_params=None):
        super(FusionFeedForward, self).__init__()
        # We set hidden_dim as a default to 1960
        self.fc1 = nn.Sequential(nn.Linear(dim, hidden_dim))
        self.fc2 = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, dim))
        assert t2t_params is not None
        self.t2t_params = t2t_params
        self.kernel_shape = reduce((lambda x, y: x * y), t2t_params['kernel_size']) # 49 my 25

    def forward(self, x, output_size):
        n_vecs = 1
        for i, d in enumerate(self.t2t_params['kernel_size']):
            n_vecs *= int((output_size[i] + 2 * self.t2t_params['padding'][i] -
                           (d - 1) - 1) / self.t2t_params['stride'][i] + 1)

        x = self.fc1(x)
        b, n, c = x.size()
        normalizer = x.new_ones(b, n, self.kernel_shape).view(-1, n_vecs, self.kernel_shape).permute(0, 2, 1)
        normalizer = F.fold(normalizer,
                            output_size=output_size,
                            kernel_size=self.t2t_params['kernel_size'],
                            padding=self.t2t_params['padding'],
                            stride=self.t2t_params['stride'])

        x = F.fold(x.view(-1, n_vecs, c).permute(0, 2, 1),
                   output_size=output_size,
                   kernel_size=self.t2t_params['kernel_size'],
                   padding=self.t2t_params['padding'],
                   stride=self.t2t_params['stride'])

        x = F.unfold(x / normalizer,
                     kernel_size=self.t2t_params['kernel_size'],
                     padding=self.t2t_params['padding'],
                     stride=self.t2t_params['stride']).permute(
                         0, 2, 1).contiguous().view(b, n, c)
        x = self.fc2(x)
        return x


def window_partition(x, window_size, n_head):
    """
    Args:
        x: shape is (B, T, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B, num_windows_h, num_windows_w, n_head, T, window_size, window_size, C//n_head)
    """
    B, T, H, W, C = x.shape
    x = x.view(B, T, H // window_size[0], window_size[0], W // window_size[1], window_size[1], n_head, C//n_head)
    windows = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return windows
def roll_with_zero_padding(input, shifts, dims):
    for dim, shift in zip(dims, shifts):
        # 获取输入张量在当前维度上的大小
        size = input.size(dim)
        
        # 计算实际的移动量
        # shift = shift % size
        
        # 使用 torch.roll 进行循环移动
        input = torch.roll(input, shift, dim)
        
        # 将超出边界的部分设为零
        if shift > 0:
            input.index_fill_(dim, torch.arange(0, shift), 0)
        elif shift < 0:
            input.index_fill_(dim, torch.arange(size+shift,size), 0)
    
    return input
class extract_overlapping_windows(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(extract_overlapping_windows, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.t2t = nn.Unfold(kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
        

    def forward(self, x, b, output_size,n_head):
        f_h = int((output_size[0] + 2 * self.padding[0] -
                   (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        f_w = int((output_size[1] + 2 * self.padding[1] -
                   (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        # b, n_wh*n_ww, self.n_head, t, w_h*w_w, c_head
        b_t,c,h,w = x.shape
        feat = self.t2t(x)
        # feat shape [b*t, ks*ks*c，num_vec ]
        feat = feat.permute(0, 2, 1)
        # feat shape [b*t, num_vec, ks*ks*c]
        # feat = self.embedding(feat)
        # feat shape after embedding [b, t*num_vec, hidden]
        feat = feat.view(b, -1, f_h*f_w, self.kernel_size[0]*self.kernel_size[1],c)
        t = feat.size(1)
        feat = feat.view(b,t,f_h*f_w,self.kernel_size[0]*self.kernel_size[1],n_head,c//n_head)
        feat = feat.permute(0,2,4,1,3,5)
        return feat

class SparseWindowAttention(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size=(4,4), qkv_bias=True, attn_drop=0., proj_drop=0., 
                pooling_token=True):
        super().__init__()
        assert dim % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(dim, dim, qkv_bias)
        self.query = nn.Linear(dim, dim, qkv_bias)
        self.value = nn.Linear(dim, dim, qkv_bias)
        # regularization
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        # output projection
        self.proj = nn.Linear(dim, dim)
        self.n_head = n_head
        self.window_size = window_size
        self.pooling_token = pooling_token
        if self.pooling_token:
            ks, stride = pool_size, pool_size
            self.pool_layer = nn.Conv2d(dim, dim, kernel_size=ks, stride=stride, padding=(0, 0), groups=dim)
            self.pool_layer.weight.data.fill_(1. / (pool_size[0] * pool_size[1]))
            self.pool_layer.bias.data.fill_(0)
        
        self.expand_size = tuple((i + 1) // 2 for i in window_size)
        
        n_wh = math.ceil(32 / self.window_size[0])
        n_ww = math.ceil(32 / self.window_size[1])
        new_h = n_wh * self.window_size[0] # 20
        new_w = n_ww * self.window_size[1] # 36
        
        
        if any(i > 0 for i in self.expand_size):
        
            # get mask for rolled k and rolled v
            mask_tl = torch.ones(self.window_size[0], self.window_size[1])
            mask_tl[:-self.expand_size[0], :-self.expand_size[1]] = 0

            mask_expand_padding_tl = roll_with_zero_padding(torch.ones(new_h,new_w),(-self.expand_size[0], -self.expand_size[1]),(0,1))

            mask_tr = torch.ones(self.window_size[0], self.window_size[1])
            mask_tr[:-self.expand_size[0], self.expand_size[1]:] = 0
            mask_expand_padding_tr = roll_with_zero_padding(torch.ones(new_h,new_w),(-self.expand_size[0], self.expand_size[1]),(0,1))


            mask_bl = torch.ones(self.window_size[0], self.window_size[1])
            mask_bl[self.expand_size[0]:, :-self.expand_size[1]] = 0
            mask_expand_padding_bl = roll_with_zero_padding(torch.ones(new_h,new_w),(self.expand_size[0], -self.expand_size[1]),(0,1))



            mask_br = torch.ones(self.window_size[0], self.window_size[1])
            mask_br[self.expand_size[0]:, self.expand_size[1]:] = 0
            mask_expand_padding_br = roll_with_zero_padding(torch.ones(new_h,new_w),(self.expand_size[0], self.expand_size[1]),(0,1))


            masrool_k = torch.stack((mask_tl, mask_tr, mask_bl, mask_br), 0).flatten(0)
            mask_padding = torch.stack((mask_expand_padding_tl,mask_expand_padding_tr,mask_expand_padding_bl,mask_expand_padding_br),0)

            self.register_buffer("valid_ind_rolled", masrool_k.nonzero(as_tuple=False).view(-1))
            self.register_buffer("mask_padding", mask_padding)

            

            
        

        
    def inint_projection(self):
        nn.init.normal_(self.key.weight.data, 0.0, 0.015)
        nn.init.normal_(self.query.weight.data, 0.0, 0.015)
        nn.init.normal_(self.value.weight.data, 0.0, 0.015)

        nn.init.constant_(self.key.bias.data, 0.0)
        nn.init.constant_(self.query.bias.data, 0.0)
        nn.init.constant_(self.value.bias.data, 0.0)


        nn.init.normal_(self.proj.weight.data, 0.0, 0.015)
        nn.init.constant_(self.proj.bias.data, 0.0)
    
    def forward(self, x, qmask,kvmask, T_ind=None, attn_mask=None):
        
        b, t, h, w, c = x.shape # 20 36
        w_h, w_w = self.window_size[0], self.window_size[1]
        c_head = c // self.n_head
        n_wh = math.ceil(h / self.window_size[0])
        n_ww = math.ceil(w / self.window_size[1])
        new_h = n_wh * self.window_size[0] # 20
        new_w = n_ww * self.window_size[1] # 36
        pad_r = new_w - w
        pad_b = new_h - h
        t_token = t//4
        # reverse order
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x,(0, 0, 0, pad_r, 0, pad_b, 0, 0), mode='constant', value=0) 
            qmask = F.pad(qmask,(0, 0, 0, pad_r, 0, pad_b, 0, 0), mode='constant', value=0) 
            kvmask = F.pad(kvmask,(0, 0, 0, pad_r, 0, pad_b, 0, 0), mode='constant', value=0)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        

        


        win_q = window_partition(q.contiguous(), self.window_size, self.n_head).view(b, n_wh*n_ww, self.n_head, t, w_h*w_w, c_head)
        win_k = window_partition(k.contiguous(), self.window_size, self.n_head).view(b, n_wh*n_ww, self.n_head, t, w_h*w_w, c_head)
        win_v = window_partition(v.contiguous(), self.window_size, self.n_head).view(b, n_wh*n_ww, self.n_head, t, w_h*w_w, c_head)
        

        
        (k_tl, v_tl) = map(lambda a: torch.roll(a, shifts=(-self.expand_size[0], -self.expand_size[1]), dims=(2, 3))*self.mask_padding[0].view(1, 1, new_h, new_w, 1), (k, v))
        (k_tr, v_tr) = map(lambda a: torch.roll(a, shifts=(-self.expand_size[0], self.expand_size[1]), dims=(2, 3))*self.mask_padding[1].view(1, 1, new_h, new_w, 1), (k, v))
        (k_bl, v_bl) = map(lambda a: torch.roll(a, shifts=(self.expand_size[0], -self.expand_size[1]), dims=(2, 3))*self.mask_padding[2].view(1, 1, new_h, new_w, 1), (k, v))
        (k_br, v_br) = map(lambda a: torch.roll(a, shifts=(self.expand_size[0], self.expand_size[1]), dims=(2, 3))*self.mask_padding[3].view(1, 1, new_h, new_w, 1), (k, v))

        (k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows) = map(
            lambda a: window_partition(a, self.window_size, self.n_head).view(b, n_wh*n_ww, self.n_head, t, w_h*w_w, c_head), 
            (k_tl, k_tr, k_bl, k_br))
        (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows) = map(
            lambda a: window_partition(a, self.window_size, self.n_head).view(b, n_wh*n_ww, self.n_head, t, w_h*w_w, c_head), 
            (v_tl, v_tr, v_bl, v_br))
        rool_k = torch.cat((k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows), 4).contiguous()
        rool_v = torch.cat((v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows), 4).contiguous() # [b, n_wh*n_ww, n_head, t, w_h*w_w, c_head]
        # mask out tokens in current window
        rool_k = rool_k[:, :, :, :, self.valid_ind_rolled]
        rool_v = rool_v[:, :, :, :, self.valid_ind_rolled]
        roll_N = rool_k.shape[4]
        rool_k = rool_k.view(b, n_wh*n_ww, self.n_head, t, roll_N, c // self.n_head)
        rool_v = rool_v.view(b, n_wh*n_ww, self.n_head, t, roll_N, c // self.n_head)
        win_k = torch.cat((win_k, rool_k), dim=4)
        win_v = torch.cat((win_v, rool_v), dim=4)
        
        
        # pool_k and pool_v
        if self.pooling_token:
            pool_x = self.pool_layer(x.view(b*t, new_h, new_w, c).permute(0,3,1,2))
            _, _, p_h, p_w = pool_x.shape
            pool_x = pool_x.permute(0,2,3,1).view(b, t, p_h, p_w, c)
            # pool_k
            pool_k = self.key(pool_x).unsqueeze(1).repeat(1, n_wh*n_ww, 1, 1, 1, 1) # [b, n_wh*n_ww, t, p_h, p_w, c]
            pool_k = pool_k.view(b, n_wh*n_ww, t, p_h, p_w, self.n_head, c_head).permute(0,1,5,2,3,4,6)
            pool_k = pool_k.contiguous().view(b, n_wh*n_ww, self.n_head, t, p_h*p_w, c_head)
            win_k = torch.cat((win_k, pool_k), dim=4)
            # pool_v
            pool_v = self.value(pool_x).unsqueeze(1).repeat(1, n_wh*n_ww, 1, 1, 1, 1) # [b, n_wh*n_ww, t, p_h, p_w, c]
            pool_v = pool_v.view(b, n_wh*n_ww, t, p_h, p_w, self.n_head, c_head).permute(0,1,5,2,3,4,6)
            pool_v = pool_v.contiguous().view(b, n_wh*n_ww, self.n_head, t, p_h*p_w, c_head)
            win_v = torch.cat((win_v, pool_v), dim=4)

        # [b, n_wh*n_ww, n_head, t, w_h*w_w, c_head]
    
        l_t = qmask.size(1)
        qmask = qmask.view(b,l_t,n_wh*n_ww).permute(0,2,1).reshape(b,n_wh*n_ww*l_t)
        kvmask = kvmask.view(b,l_t,n_wh*n_ww).permute(0,2,1).reshape(b,n_wh*n_ww*l_t)
        # b, n_wh*n_ww, self.n_head, t, w_h*w_w, c_head
        twin_q = win_q.permute(0,1,3,2,4,5).reshape(b, n_wh*n_ww*t, self.n_head, w_h*w_w, c_head)
        twin_k = win_k.reshape(b, n_wh*n_ww, self.n_head, t,-1, c_head)
        twin_v = win_v.reshape(b, n_wh*n_ww, self.n_head, t,-1, c_head)
        twin_k = twin_k.permute(0,1,3,2,4,5).reshape(b, n_wh*n_ww*t, self.n_head,-1, c_head)
        twin_v = twin_v.permute(0,1,3,2,4,5).reshape(b, n_wh*n_ww*t, self.n_head,-1, c_head)
        out = torch.zeros_like(twin_q)
        for i in range(win_q.shape[0]):
            ### For masked windows
            mask_ind_i = qmask[i].nonzero(as_tuple=False).view(-1)
            kvmask_ind_i = kvmask[i].nonzero(as_tuple=False).view(-1)
            # mask out quary in current window
            # [b, n_wh*n_ww, n_head, t, w_h*w_w, c_head]
            mask_n = len(mask_ind_i)

            if mask_n > 0:
                # b, n_wh*n_ww, self.n_head, t, w_h*w_w, c_head
                win_q_t = twin_q[i, mask_ind_i]
                win_k_t = twin_k[i, kvmask_ind_i]
                win_v_t = twin_v[i, kvmask_ind_i]
                # mask out key and value
                win_k_t = win_k_t.view(mask_n, self.n_head, -1, c_head)
                win_v_t = win_v_t.view(mask_n, self.n_head, -1, c_head)
                kv_win_token = win_v_t.shape[-2]
                win_k_t = win_k_t.view(-1,t_token,self.n_head,kv_win_token,c_head)
                win_v_t = win_v_t.view(-1,t_token,self.n_head,kv_win_token,c_head)
                n_win = win_v_t.shape[0]
                win_k_t = win_k_t.permute(0,2,1,3,4).reshape(n_win,self.n_head,(t_token)*kv_win_token,c_head)
                win_v_t = win_v_t.permute(0,2,1,3,4).reshape(n_win,self.n_head,(t_token)*kv_win_token,c_head)

                win_q_t = win_q_t.view(n_win,t_token, self.n_head, w_h*w_w, c_head)
                win_q_t = win_q_t.permute(0,2,1,3,4).reshape(n_win,self.n_head,(t_token)*w_h*w_w,c_head)


                att_t = (win_q_t @ win_k_t.transpose(-2, -1)) * (1.0 / math.sqrt(win_q_t.size(-1)))
                att_t = F.softmax(att_t, dim=-1)
                att_t = self.attn_drop(att_t)
                y_t = att_t @ win_v_t 
                y_t = y_t.view(n_win,self.n_head, (t_token),w_h*w_w, c_head)
                y_t = y_t.permute(0,2,1,3,4).reshape(n_win*(t_token),self.n_head, w_h*w_w, c_head)
                
                out[i,mask_ind_i] = y_t.view(-1, self.n_head, w_h*w_w, c_head)
            
            ### For unmasked windows
            unmask_ind_i = (1-qmask[i]).nonzero(as_tuple=False).view(-1)
            # [b, n_wh*n_ww*t, self.n_head,-1, c_head]
            win_q_s = twin_q[i, unmask_ind_i]
            win_k_s = twin_k[i, unmask_ind_i, :, :w_h*w_w].view(len(unmask_ind_i), self.n_head, w_h*w_w, c_head)
            win_v_s = twin_v[i, unmask_ind_i, :, :w_h*w_w].view(len(unmask_ind_i), self.n_head, w_h*w_w, c_head)
            
            win_q_s = win_q_s.view(-1,t_token,self.n_head, w_h*w_w, c_head)
            n_un_token = win_q_s.shape[0]
            win_q_s = win_q_s.permute(0,2,1,3,4).reshape(n_un_token,self.n_head, (t_token)*w_h*w_w, c_head)

            win_k_s = win_k_s.view(-1,t_token,self.n_head, w_h*w_w, c_head)
            win_k_s = win_k_s.permute(0,2,1,3,4).reshape(n_un_token,self.n_head, (t_token)*w_h*w_w, c_head)

            win_v_s = win_v_s.view(-1,t_token,self.n_head, w_h*w_w, c_head)
            win_v_s = win_v_s.permute(0,2,1,3,4).reshape(n_un_token,self.n_head, (t_token)*w_h*w_w, c_head)


            att_s = (win_q_s @ win_k_s.transpose(-2, -1)) * (1.0 / math.sqrt(win_q_s.size(-1)))
            att_s = F.softmax(att_s, dim=-1)
            att_s = self.attn_drop(att_s)
            y_s = att_s @ win_v_s
            y_s = y_s.view(n_un_token,self.n_head, (t_token),w_h*w_w, c_head)
            y_s = y_s.permute(0,2,1,3,4).reshape(n_un_token*(t_token),self.n_head, w_h*w_w, c_head)
            out[i, unmask_ind_i] = y_s
            
        
        # 1, 512, 4, 16, 128
        # re-assemble all head outputs side by side
        out = out.view(b, n_wh, n_ww, t, self.n_head,  w_h, w_w, c_head)
        # b,t,n_wh,w_h,n_ww,w_w,n_head,c_head
        # b,t,n_wh, w_h,n_ww,w_w,n_head,c_head
        out = out.permute(0, 3, 1, 5, 2, 6, 4, 7).contiguous().view(b, t, new_h, new_w, c)


        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :h, :w, :]
        
        # output projection
        out = self.proj_drop(self.proj(out))
        return out





class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6,weight=1):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)*weight))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
class TemporalSparseTransformer(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size,
                norm_layer=nn.LayerNorm, t2t_params=None):
        super().__init__()
        self.window_size = window_size
        self.attention = SparseWindowAttention(dim, n_head, window_size, pool_size)
        
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = FusionFeedForward(dim, t2t_params=t2t_params)
        

    def forward(self, x, fold_x_size, qmask,kvmask, T_ind=None):
        """
        Args:
            x: image tokens, shape [B T H W C]
            fold_x_size: fold feature size, shape [60 108]
            mask: mask tokens, shape [B T H W 1]
        Returns:
            out_tokens: shape [B T H W C]
        """
        B, T, H, W, C = x.shape # 20 36

        shortcut = x
        x = self.norm1(x)
        
        att_x = self.attention(x, qmask,kvmask, T_ind)
        

        # FFN
        x = shortcut + att_x
        y = self.norm2(x)
        x = x + self.mlp(y.view(B, T * H * W, C), fold_x_size).view(B, T, H, W, C)

        return x


class TemporalSparseTransformerBlock(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size, depths, t2t_params=None,gain=0.02):
        super().__init__()
        blocks = []
        for i in range(depths):
             blocks.append(
                TemporalSparseTransformer(dim, n_head, window_size, pool_size, t2t_params=t2t_params)
             )
        self.transformer = nn.Sequential(*blocks)
        self.depths = depths
        self.init_weights(gain=gain)
    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            print(m)
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1
                                           or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' %
                        init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)
    def forward(self, x, fold_x_size, qmasks,kvmasks, t_dilation=2):
        """
        Args:
            x: image tokens, shape [B T H W C]
            fold_x_size: fold feature size, shape [60 108]
            l_mask: local mask tokens, shape [B T H W 1]
        Returns:
            out_tokens: shape [B T H W C]
        """
        assert self.depths % t_dilation == 0, 'wrong t_dilation input.'
        T = x.size(1)
        T_ind = [torch.arange(i, T, t_dilation) for i in range(t_dilation)] * (self.depths // t_dilation)

        for i in range(0, self.depths):
            x = self.transformer[i](x, fold_x_size, qmasks[i],kvmasks[i], T_ind[i])
        return x
