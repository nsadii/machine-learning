clc;clear;close all;warning off;

data = xlsread ('emglatih2.xlsx',1);

% Proses Normalisasi Data
max_data = max(max(data));
min_data = min(min(data));
 
[m,n] = size(data);
data_norm = zeros(m,n);
for x = 1:m
    for y = 1:n
        data_norm(x,y) = (data(x,y)-min_data)*(1)/(max_data-min_data)+0;
    end
end

data_latih = data_norm;
data_target = zeros (m,n);


for b = 1:n-1
    for a = 1:1
    data_target(a,b) = data_norm(b+a);
    end
end

% Pembuatan JST
net = newff(minmax(data_latih),[2 16 1],{'logsig','logsig','logsig'},'traingdx');
 
% Memberikan nilai untuk mempengaruhi proses pelatihan
net.performFcn = 'mse';
net.trainParam.goal = 0.0001;
net.trainParam.show = 20;
net.trainParam.epochs = 100;
net.trainParam.mc = 0.95;
net.trainParam.lr = 0.1;
 
% Proses training
[net_keluaran,tr,Y,E] = train(net,data_latih,data_target);
 
% Hasil setelah pelatihan
bobot_hidden = net_keluaran.IW{1,1};
bobot_keluaran = net_keluaran.LW{2,1};
bias_hidden = net_keluaran.b{1,1};
bias_keluaran = net_keluaran.b{2,1};
jumlah_iterasi = tr.num_epochs;
nilai_keluaran = Y;
nilai_error = E;
error_MSE = (1/n)*sum(nilai_error.^2);
 
save net.mat net_keluaran
 
% Hasil prediksi
hasil_latih = sim(net_keluaran,data_latih);

figure,
plotregression(data_target,hasil_latih,'Regression')
 
figure,
plotperform(tr)
 
peak=(data_target+0.1)>hasil_latih;
figure,
plot(hasil_latih,'b-')
hold on
plot(data_target,'r-')
plot(peak,'k-')
hold off

grid on
title(strcat(['Grafik Keluaran JST vs Target dengan nilai MSE = ',...
num2str(error_MSE)]))
xlabel('Pola ke-')
ylabel('')
legend('Keluaran JST','Target','Location','Best')

load net.mat

uji = xlsread ('emglatih2.xlsx',2);

% Proses Normalisasi Data
max_uji = max(max(uji));
min_uji = min(min(uji));
 
[m,n] = size(uji);
uji_norm = zeros(m,n);
for x = 1:m
    for y = 1:n
        uji_norm(x,y) = (uji(x,y)-min_uji)*(1)/(max_uji-min_uji)+0;
    end
end

uji_latih = uji_norm;
uji_target = zeros (m,n);


for b = 1:n-1
    for a = 1:1
    uji_target(a,b) = uji_norm(b+a);
    end
end

hasil_uji = sim(net_keluaran,uji_latih);
nilai_error = hasil_uji-uji_target;

error_MSE = (1/n)*sum(nilai_error.^2);

peak=(uji_target+0.1)>hasil_uji;
figure,
plot(hasil_uji,'b-')
hold on
plot(uji_target,'r-')
plot(peak,'k-')
hold off

grid on
title(strcat(['Grafik Keluaran JST vs Target dengan nilai MSE = ',...
num2str(error_MSE)]))
xlabel('Pola ke-')
ylabel('')
legend('Keluaran JST','Target','Location','Best')