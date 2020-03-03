function [F_CB,all_beams]=UPA_codebook_generator(Mx,My,Mz,over_sampling_x,over_sampling_y,over_sampling_z,ant_spacing)

% Code by A. Alkhateeb et al.
% Commented by M. G. Khafagy

% INPUT: 
% 1) Number of antennas in (x,y,z) dimensions, 
% 2) Oversampling in (x,y,z),
% 3) antenna spacing (to calculate the constant kd).

% OUTPUT: 
% 1) F_CB:          3D Codebook
% 2) all_beams:     3D Indices

% Constant
kd=2*pi*ant_spacing;    

% Index Vectors
antx_index=0:1:Mx-1;
anty_index=0:1:My-1;
antz_index=0:1:Mz-1;

% M=Mx*My*Mz; % Total number of antennas in the UPA

% Defining the RF beamforming codebook
codebook_size_x=over_sampling_x*Mx;
codebook_size_y=over_sampling_y*My;
codebook_size_z=over_sampling_z*Mz;

% ============= X Axis =============
theta_qx=0:pi/codebook_size_x:pi-1e-6; % quantized beamsteering angles
% Why not theta_qx=0:pi/codebook_size_x:pi-pi/codebook_size_x 
% It assumes pi/codebook_size_x will be always greater than 1e-6 .. ok
F_CBx=zeros(Mx,codebook_size_x);
for i=1:1:length(theta_qx)              % For each beamsteering angle in the x directon
    F_CBx(:,i)=sqrt(1/Mx)*exp(-1j*kd*antx_index'*cos(theta_qx(i)));     % calculate the reflection vector in x direction
end
% ============= Y Axis =============
range_y=(20+307)*pi/180;
theta_qy=20*pi/180:-range_y/codebook_size_y:-307*pi/180+1e-6; % quantized beamsteering angles
F_CBy=zeros(My,codebook_size_y);
for i=1:1:length(theta_qy)              % For each beamsteering angle in the y directon
    F_CBy(:,i)=sqrt(1/My)*exp(-1j*anty_index'*theta_qy(i)); % calculate the reflection vector in y direction 
    % ##############################################################
    % ###################### WHY NO kd HERE ########################
    % ################### DIFFERENT CALCULATION ####################
end
% ============= Z Axis =============
theta_qz=0:pi/codebook_size_z:pi-1e-6; % quantized beamsteering angles
F_CBz=zeros(Mz,codebook_size_z);
for i=1:1:length(theta_qz)              % For each beamsteering angle in the z directon
    F_CBz(:,i)=sqrt(1/Mz)*exp(-1j*kd*antz_index'*cos(theta_qz(i)));     % calculate the reflection vector in z direction
end

% ============= 3D codebook =============
F_CBxy=kron(F_CBy,F_CBx);
F_CB=kron(F_CBz,F_CBxy);


% ============= 3D Indices =============
beams_x=1:1:codebook_size_x;
beams_y=1:1:codebook_size_y;
beams_z=1:1:codebook_size_z;

Mxx_Ind=repmat(beams_x,1,codebook_size_y*codebook_size_z)';
Myy_Ind=repmat(reshape(repmat(beams_y,codebook_size_x,1),1,codebook_size_x*codebook_size_y),1,codebook_size_z)';
Mzz_Ind=reshape(repmat(beams_z,codebook_size_x*codebook_size_y,1),1,codebook_size_x*codebook_size_y*codebook_size_z)';

Tx=cat(3,Mxx_Ind',Myy_Ind',Mzz_Ind');
all_beams=reshape(Tx,[],3);
end
