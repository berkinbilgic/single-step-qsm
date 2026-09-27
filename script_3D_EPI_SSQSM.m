%% load data

load EPI_3D_magn
load EPI_3D_mask_bet
load EPI_3D_phs

mosaic(imrotate(magn(40:end-40,10:end-10,65:4:end) .* mask_bet(40:end-40,10:end-10,65:4:end), 90), 3, 5, 1, 'Magnitude', [0,.6])
mosaic(imrotate(phs(40:end-40,10:end-10,65:4:end) .* mask_bet(40:end-40,10:end-10,65:4:end), 90), 3, 5, 2, 'Phase', [-pi,pi])


TE = 21e-3;           % second  
B0 = 2.89;            % Tesla
gyro = 2*pi*42.58;
N = size(mask_bet);


%% Laplacian unwrapping kernel

tic
    ksize = [3, 3, 3];               
    khsize = (ksize-1)/2;

    kernel = [];

    kernel(:,:,1) = [0 0 0; 0 1 0; 0 0 0];
    kernel(:,:,2) = [0 1 0; 1 -6 1; 0 1 0];
    kernel(:,:,3) = [0 0 0; 0 1 0; 0 0 0];

    Kernel = zeros(N);
    Kernel( 1+N(1)/2 - khsize(1) : 1+N(1)/2 + khsize(1), 1+N(2)/2 - khsize(2) : 1+N(2)/2 + khsize(2), 1+N(3)/2 - khsize(3) : 1+N(3)/2 + khsize(3) ) = -kernel;

    del_op = fftn(fftshift(Kernel));
    del_inv = zeros(size(del_op));
    del_inv( del_op~=0 ) = 1 ./ del_op( del_op~=0 );
    
    exp_phase = exp(1i.*phs);
    Phase_unwrap = del_inv .* fftn(imag(conj(exp_phase) .* ifftn(del_op .* fftn(exp_phase))));
toc


%% V-Sharp filtering for background removal

tic 

stol = .2;                  % truncation threshold
Kernel_Sizes = 11:-2:3;

DiffMask = zeros([N, length(Kernel_Sizes)]);
Mask_Sharp = zeros([N, length(Kernel_Sizes)]);
Del_Sharp = zeros([N, length(Kernel_Sizes)]);

phs = 0;

for k = 1:length(Kernel_Sizes)
    Kernel_Size = Kernel_Sizes(k);

    ksize = [Kernel_Size, Kernel_Size, Kernel_Size];                % Sharp kernel size

    khsize = (ksize-1)/2;
    [a,b,c] = meshgrid(-khsize(2):khsize(2), -khsize(1):khsize(1), -khsize(3):khsize(3));

    kernel = (a.^2 / khsize(1)^2 + b.^2 / khsize(2)^2 + c.^2 / khsize(3)^2 ) <= 1;
    kernel = -kernel / sum(kernel(:));
    kernel(khsize(1)+1,khsize(2)+1,khsize(3)+1) = 1 + kernel(khsize(1)+1,khsize(2)+1,khsize(3)+1);

    Kernel = zeros(N);
    Kernel( 1+N(1)/2 - khsize(1) : 1+N(1)/2 + khsize(1), 1+N(2)/2 - khsize(2) : 1+N(2)/2 + khsize(2), 1+N(3)/2 - khsize(3) : 1+N(3)/2 + khsize(3) ) = kernel;

    del_sharp = fftn(fftshift(Kernel));    

    % erode mask to remove convolution artifacts
    erode_size = ksize + 1;

    msk_sharp = imerode(mask_bet, strel('line', erode_size(1), 0));
    msk_sharp = imerode(msk_sharp, strel('line', erode_size(2), 90));
    msk_sharp = permute(msk_sharp, [1,3,2]);
    msk_sharp = imerode(msk_sharp, strel('line', erode_size(3), 0));
    msk_sharp = permute(msk_sharp, [1,3,2]);

    Mask_Sharp(:,:,:,k) = msk_sharp; 
    Del_Sharp(:,:,:,k) = del_sharp; 
    
    if k == 1
        DiffMask(:,:,:,1) = Mask_Sharp(:,:,:,1);
    else
        DiffMask(:,:,:,k) = Mask_Sharp(:,:,:,k) - Mask_Sharp(:,:,:,k-1);
    end
    
    phs = phs + DiffMask(:,:,:,k) .* ifftn(Del_Sharp(:,:,:,k) .* Phase_unwrap) / (TE * B0 * gyro);
end

del_sharp = Del_Sharp(:,:,:,1);         % first kernel
mask_sharp = Mask_Sharp(:,:,:,end);     % largest mask

inv_sharp = zeros(size(del_sharp));
inv_sharp( abs(del_sharp)>stol ) = 1 ./ del_sharp( abs(del_sharp)>stol );

phase_sharp = mask_sharp .* ifftn(inv_sharp .* fftn(phs));

toc

mosaic(-imrotate(phase_sharp(50:end-50,20:end-25,70:5:end,:), 90), 2, 5, 1, 'V-Sharp Phase', [-.04,.04] )


%% gradient masks from magnitude image using k-space gradients

[k2,k1,k3] = meshgrid(0:N(2)-1, 0:N(1)-1, 0:N(3)-1);
fdx = -1 + exp(2*pi*1i*k1/N(1));
fdy = -1 + exp(2*pi*1i*k2/N(2));
fdz = -1 + exp(2*pi*1i*k3/N(3));

cfdx = conj(fdx);       cfdy = conj(fdy);       cfdz = conj(fdz);

magni = magn .* mask_sharp;

Magn = fftn(magni / max(magni(:)));
magn_grad = cat(4, ifftn(Magn.*fdx), ifftn(Magn.*fdy), ifftn(Magn.*fdz));

magn_weight = zeros(size(magn_grad));

for s = 1:size(magn_grad,4)
    magn_use = abs(magn_grad(:,:,:,s));
    
    magn_order = sort(magn_use(mask_sharp==1), 'descend');
    magn_threshold = magn_order( round(length(magn_order) * .15) );
    magn_weight(:,:,:,s) = magn_use <= magn_threshold;
end

mosaic(imrotate(sqrt(sum(abs(magn_weight(50:end-50,20:end-25,70:5:end,:)).^2,4)), 90), 2, 5, 5, 'Edge Mask', [0,sqrt(3)] )


%% Closed-Form L2-regularized QSM 

lambda_L2 = 2.9e-2;     % L2-regularization parameter selected with L-curve

FOV = N .* [1, 1, 1];   % Field of view(in milimeters)
D = fftshift(kspace_kernel(FOV, N));        % k-space kernel
D2 = abs(D).^2;
E2 = abs(fdx).^2 + abs(fdy).^2 + abs(fdz).^2;

chi_cfL2 = mask_sharp .* ifftn(conj(D) .* fftn(phase_sharp) ./ (eps + D2 + lambda_L2*E2));

mosaic(imrotate(chi_cfL2(50:end-50,20:end-25,70:5:end), 90), 2, 5, 2, 'CF L2', [-.14,.14] )


%% L2-regularized QSM with Magnitude Weighting

rhs = conj(D) .* fftn(phase_sharp);         % right hand side      
A_inv = 1 ./ (eps + D2 + lambda_L2*E2);     % preconditioner
Fchi0 = rhs .* A_inv;                       % initial guess

precond_inverse = @(x, A_inv) A_inv(:).*x;

tic
    [F_chi, flag, pcg_res, pcg_iter] = pcg(@(x)CFL2_Edge(x, D2, lambda_L2, fdx, fdy, fdz, cfdx, cfdy, cfdz, magn_weight), ...
        rhs(:), 1e-2, 30, @(x)precond_inverse(x, A_inv), [], Fchi0(:));
toc

disp(['PCG iter: ', num2str(pcg_iter), '   PCG residual: ', num2str(pcg_res)])

chi_cfL2e = mask_sharp .* real(ifftn(reshape(F_chi, N)));

mosaic(imrotate(chi_cfL2e(50:end-50,20:end-25,70:5:end), 90), 2, 5, 3, 'CF L2 Edge', [-.14,.14] )


%% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%% Single-step QSM with V-Sharp
%% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Rhs_pcg = 0;
for k = 1:size(Del_Sharp,4)
    Rhs_pcg = Rhs_pcg + conj(Del_Sharp(:,:,:,k)) .* fftn(DiffMask(:,:,:,k) .* phs);
end
Rhs_pcg = conj(D) .* Rhs_pcg;       % right hand side


lambda_ss = 2.9e-2;                 % regularization parameter

B_inv = 1 ./ (eps + abs(Del_Sharp(:,:,:,1) .* D).^2 + lambda_ss*E2);        % preconditioner
precond_inv = @(x, B_inv) B_inv(:).*x;

x0 = B_inv .* Rhs_pcg;              % initial guess


tic
    [F_chi, flag, pcg_res, pcg_iter] = pcg(@(x)SSQSM_vsharp(x, D, Del_Sharp, conj(Del_Sharp), DiffMask, lambda_ss, fdx, fdy, fdz, cfdx, cfdy, cfdz, magn_weight), ...
        Rhs_pcg(:), 1e-2, 30, @(x) precond_inv(x, B_inv), [], x0(:));
toc

disp(['PCG iter: ', num2str(pcg_iter), '   PCG residual: ', num2str(pcg_res)])

chi_ss = real(ifftn(reshape(F_chi, N))) .* mask_sharp;

mosaic(imrotate(chi_ss(50:end-50,20:end-25,70:5:end),90), 2, 5, 4, 'ssqsm v-sharp', [-.14,.14])


%% Display views in 3 planes

x_pos = 128;
y_pos = 116;
z_pos = 78;

% L2-regularized QSM
chi_disp = chi_cfL2e;
chi_disp(chi_disp==0) = -1;

fig_no = 11;
scale_fig = [-.1,.13];

figure(fig_no), subplot(131), imagesc( imrotate(max(chi_disp(45:end-45,20:end-25,z_pos-1:z_pos+1),[],3), 90), scale_fig ), axis image off, colormap gray
figure(fig_no), subplot(132), imagesc( imrotate(squeeze(max(chi_disp(45:end-45,y_pos-1:y_pos+1,35:end-15),[],2)), 90), scale_fig ), axis image off, colormap gray
figure(fig_no), subplot(133), imagesc( fliplr(imrotate(squeeze(max(chi_disp(x_pos-1:x_pos+1,20:end-20,25:end-15),[],1)), 90)), scale_fig ), axis image off, colormap gray
set(gcf, 'Color', [0 0 0]);


% Single-Step QSM
chi_disp = chi_ss;
chi_disp(chi_disp==0) = -1;

fig_no = 12;

figure(fig_no), subplot(131), imagesc( imrotate(max(chi_disp(45:end-45,20:end-25,z_pos-1:z_pos+1),[],3), 90), scale_fig ), axis image off, colormap gray
figure(fig_no), subplot(132), imagesc( imrotate(squeeze(max(chi_disp(45:end-45,y_pos-1:y_pos+1,35:end-15),[],2)), 90), scale_fig ), axis image off, colormap gray
figure(fig_no), subplot(133), imagesc( fliplr(imrotate(squeeze(max(chi_disp(x_pos-1:x_pos+1,20:end-20,25:end-15),[],1)), 90)), scale_fig ), axis image off, colormap gray
set(gcf, 'Color', [0 0 0]);


% Tissue phase
phs_disp = phase_sharp;
phs_disp(phs_disp==0) = 1;

fig_no = 13;
scale_fig = [-.035,.035];
figure(fig_no), subplot(131), imagesc( -imrotate(mean(phs_disp(45:end-45,20:end-25,z_pos-1:z_pos+1),3), 90), scale_fig ), axis image off, colormap gray
figure(fig_no), subplot(132), imagesc( -imrotate(squeeze(mean(phs_disp(45:end-45,y_pos-1:y_pos+1,35:end-15),2)), 90), scale_fig ), axis image off, colormap gray
figure(fig_no), subplot(133), imagesc( -fliplr(imrotate(squeeze(mean(phs_disp(x_pos-1:x_pos+1,20:end-20,25:end-15),1)), 90)), scale_fig ), axis image off, colormap gray
set(gcf, 'Color', [0 0 0]);


