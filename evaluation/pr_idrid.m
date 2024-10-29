clc
clear
pred_path='../results/';
label_path= '/home/data/guo/dataset/IDRiD/label_raw/';
num_threshold = 30;

%%%% EX %%%%
pred_img_all=dir([pred_path '*_ex.png']);
re_array=zeros(num_threshold,1);
pr_array=zeros(num_threshold,1);

fprintf("Evaluating EX...\n")
k=0;
for i=0:num_threshold % loop over thresholds
    thres = i/num_threshold;
    sum_tp = 0;
		sum_tn = 0;
    sum_fp = 0;
    sum_fn = 0;

    for ind_img=1:length(pred_img_all)
        tp = 0;
		    tn = 0;
        fp = 0;
        fn = 0;
        pred_img = imread([pred_path pred_img_all(ind_img).name]); % 0-255

      	label  = imread([label_path pred_img_all(ind_img).name(1:end-7) '_EX.tif'])/255; % 0, 1

    		if size(pred_img,1) ~= size(label, 1)
					pred_img = imresize(pred_img, [size(label,1), size(label,2)]);
				end


        pred_img = double(pred_img);
        pred_img = pred_img / 255; % 0 to 1

        pred_binary = pred_img >= thres;

        tp = sum(sum(pred_binary&label));
        fp = sum(sum(pred_binary))-tp;
        fn = sum(sum(label))-tp;

				sum_tp = sum_tp + tp;
				sum_fp = sum_fp + fp;
				sum_fn = sum_fn + fn;

			end


    if sum_tp+sum_fp == 0 || sum_tp+sum_fn == 0
        continue;
    end
    re = (sum_tp/(sum_tp+sum_fn));
    pr = (sum_tp/(sum_tp+sum_fp));

    if re == 0 && pr == 0
        continue;
    end
    k = k+1;
    re_array(k,1)=re;
    pr_array(k,1)=pr;
		f1=2*re*pr/(re+pr);
		if i==num_threshold/2
			f1_05_ex = f1;
			iou_ex = sum_tp/(sum_tp+sum_fp+sum_fn);
		end
end
re_array=re_array(1:k,1);
pr_array=pr_array(1:k,1);
re_array=fliplr(re_array')';
pr_array=fliplr(pr_array')';
re_array=[0;re_array];
pr_array=[1;pr_array];
auc_pr_ex=trapz(re_array,pr_array);



%%%%% HE %%%%
fprintf("Evaluating HE...\n")
pred_img_all=dir([pred_path '*_he.png']);
re_array=zeros(num_threshold,1);
pr_array=zeros(num_threshold,1);

k=0;
for i=0:num_threshold % loop over thresholds
    thres = i/num_threshold;
    sum_tp = 0;
		sum_tn = 0;
    sum_fp = 0;
    sum_fn = 0;

    for ind_img=1:length(pred_img_all)
        tp = 0;
		    tn = 0;
        fp = 0;
        fn = 0;
        pred_img = imread([pred_path pred_img_all(ind_img).name]); % 0-255
      	label  = imread([label_path pred_img_all(ind_img).name(1:end-7) '_HE.tif'])/255; % 0, 1

    		if size(pred_img,1) ~= size(label, 1)
					pred_img = imresize(pred_img, [size(label,1), size(label,2)]);
				end


        pred_img = double(pred_img);
        pred_img = pred_img / 255; % 0 to 1

        pred_binary = pred_img >= thres;

        tp = sum(sum(pred_binary&label));
        fp = sum(sum(pred_binary))-tp;
        fn = sum(sum(label))-tp;

				sum_tp = sum_tp + tp;
				sum_fp = sum_fp + fp;
				sum_fn = sum_fn + fn;

			end


    if sum_tp+sum_fp == 0 || sum_tp+sum_fn == 0
        continue;
    end
    re = (sum_tp/(sum_tp+sum_fn));
    pr = (sum_tp/(sum_tp+sum_fp));

    if re == 0 && pr == 0
        continue;
    end
    k = k+1;
    re_array(k,1)=re;
    pr_array(k,1)=pr;
		f1=2*re*pr/(re+pr);
		if i==num_threshold/2
			f1_05_he = f1;
			iou_he = sum_tp/(sum_tp+sum_fp+sum_fn);
		end
end
re_array=re_array(1:k,1);
pr_array=pr_array(1:k,1);
re_array=fliplr(re_array')';
pr_array=fliplr(pr_array')';
re_array=[0;re_array];
pr_array=[1;pr_array];
auc_pr_he=trapz(re_array,pr_array);

%%%%% MA %%%%
fprintf("Evaluating MA...\n")
pred_img_all=dir([pred_path '*_ma.png']);
re_array=zeros(num_threshold,1);
pr_array=zeros(num_threshold,1);

k=0;
for i=0:num_threshold % loop over thresholds
    thres = i/num_threshold;
    sum_tp = 0;
		sum_tn = 0;
    sum_fp = 0;
    sum_fn = 0;

    for ind_img=1:length(pred_img_all)
        tp = 0;
		    tn = 0;
        fp = 0;
        fn = 0;
        pred_img = imread([pred_path pred_img_all(ind_img).name]); % 0-255
      	label  = imread([label_path pred_img_all(ind_img).name(1:end-7) '_MA.tif'])/255; % 0, 1

    		if size(pred_img,1) ~= size(label, 1)
					pred_img = imresize(pred_img, [size(label,1), size(label,2)]);
				end


        pred_img = double(pred_img);
        pred_img = pred_img / 255; % 0 to 1

        pred_binary = pred_img >= thres;

        tp = sum(sum(pred_binary&label));
        fp = sum(sum(pred_binary))-tp;
        fn = sum(sum(label))-tp;

				sum_tp = sum_tp + tp;
				sum_fp = sum_fp + fp;
				sum_fn = sum_fn + fn;

			end


    if sum_tp+sum_fp == 0 || sum_tp+sum_fn == 0
			  f1_05_ma = 0;
				iou_ma = 0;
        continue;
    end
    re = (sum_tp/(sum_tp+sum_fn));
    pr = (sum_tp/(sum_tp+sum_fp));

    if re == 0 && pr == 0
        continue;
    end
    k = k+1;
    re_array(k,1)=re;
    pr_array(k,1)=pr;
		f1=2*re*pr/(re+pr);
		if i==num_threshold/2
			f1_05_ma = f1;
			iou_ma = sum_tp/(sum_tp+sum_fp+sum_fn);
		end
end
re_array=re_array(1:k,1);
pr_array=pr_array(1:k,1);
re_array=fliplr(re_array')';
pr_array=fliplr(pr_array')';
re_array=[0;re_array];
pr_array=[1;pr_array];
auc_pr_ma=trapz(re_array,pr_array);


%%%%% SE %%%%
fprintf("Evaluating SE...\n")
pred_img_all=dir([pred_path '*_se.png']);
re_array=zeros(num_threshold,1);
pr_array=zeros(num_threshold,1);

k=0;
for i=0:num_threshold % loop over thresholds
    thres = i/num_threshold;
    sum_tp = 0;
		sum_tn = 0;
    sum_fp = 0;
    sum_fn = 0;

    for ind_img=1:length(pred_img_all)
        tp = 0;
		    tn = 0;
        fp = 0;
        fn = 0;
        pred_img = imread([pred_path pred_img_all(ind_img).name]); % 0-255
      	label  = imread([label_path pred_img_all(ind_img).name(1:end-7) '_SE.tif'])/255; % 0, 1

    		if size(pred_img,1) ~= size(label, 1)
					pred_img = imresize(pred_img, [size(label,1), size(label,2)]);
				end


        pred_img = double(pred_img);
        pred_img = pred_img / 255; % 0 to 1

        pred_binary = pred_img >= thres;

        tp = sum(sum(pred_binary&label));
        fp = sum(sum(pred_binary))-tp;
        fn = sum(sum(label))-tp;

				sum_tp = sum_tp + tp;
				sum_fp = sum_fp + fp;
				sum_fn = sum_fn + fn;

			end


    if sum_tp+sum_fp == 0 || sum_tp+sum_fn == 0
        continue;
    end
    re = (sum_tp/(sum_tp+sum_fn));
    pr = (sum_tp/(sum_tp+sum_fp));

    if re == 0 && pr == 0
        continue;
    end
    k = k+1;
    re_array(k,1)=re;
    pr_array(k,1)=pr;
		f1=2*re*pr/(re+pr);
		if i==num_threshold/2
			f1_05_se = f1;
			iou_se = sum_tp/(sum_tp+sum_fp+sum_fn);
		end
end
re_array=re_array(1:k,1);
pr_array=pr_array(1:k,1);
re_array=fliplr(re_array')';
pr_array=fliplr(pr_array')';
re_array=[0;re_array];
pr_array=[1;pr_array];
auc_pr_se=trapz(re_array,pr_array);
avg_f1 = (f1_05_ex+f1_05_he+f1_05_ma+f1_05_se)/4;
avg_iou = (iou_ex+iou_he+iou_ma+iou_se)/4;
avg_pr = (auc_pr_ex+auc_pr_he+auc_pr_ma+auc_pr_se)/4;
fprintf("F1_EX:%.4f IoU_EX:%.4f PR_EX:%.4f\n", f1_05_ex, iou_ex, auc_pr_ex);
fprintf("F1_HE:%.4f IoU_HE:%.4f PR_HE:%.4f\n", f1_05_he, iou_he, auc_pr_he);
fprintf("F1_MA:%.4f IoU_MA:%.4f PR_MA:%.4f\n", f1_05_ma, iou_ma, auc_pr_ma);
fprintf("F1_SE:%.4f IoU_SE:%.4f PR_SE:%.4f\n", f1_05_se, iou_se, auc_pr_se);
fprintf("mF1:%.4f mIoU:%.4f mPR:%.4f\n", avg_f1, avg_iou, avg_pr);
