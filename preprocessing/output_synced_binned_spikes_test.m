filename = 'H5903.005_P_S.plx';

%get spike counts (and other counts) for each channel and unit
[tscounts, wfcounts, evcounts, slowcounts] = plx_info(filename,1);

% Load the PLX file and extract waveforms + spike times from best
% channel/unit pair (19, 2)
[n, ts] = plx_ts(filename, 18, 1);

% Load the photocell response
% ad49 is photocell, other weird things 50-52 ish, spikes seem to be 17-20
[adfreq, nphot, tsphot, fn, ad] = plx_ad_v(filename, 'ad49');

% Load the csm file (contains the dataRecord structure with the movie names
% for each trial)
load('H5903.005.mat');


% Analysis of the photocell ad channel (finds start times and end times of
% 'droughts', places where the response is almost zero)
droughtstarts = [];
droughtends = [];

droughtlength = 0;
droughtstart = -1;

for i = 1:numel(ad)
    if ad(i) < 0.25
        if droughtstart == -1
            droughtstart = i;
            droughtlength = droughtlength + 1;
        else
            droughtlength = droughtlength + 1;
        end
    else
        if droughtlength >= 800
            droughtstarts(end+1) = droughtstart;
            droughtends(end+1) = i;
            
        end
        droughtstart = -1;
        droughtlength = 0;
    end
end

% it seems that the white sections (constant value 1) are 5 seconds long
% and correspond to the places where the stimuli are shown
% 
% the sequence starts with a grey section and ends with a grey section, so
% there are an even number of droughts, at odd positions they immediately
% precede a white section and at even positions they immediately succeed a
% white section

%finding the white sections by looking between droughts and seeing if there
%is a point high enough

properintervals = [];

for i = 1:numel(droughtends)-1
    m = max(ad(droughtends(i):droughtstarts(i+1)));
    if m > 0.8
        properintervals = [properintervals; [droughtends(i), droughtstarts(i+1)]];
    end
end

%properintervals is a ~300 by 2 matrix (now in seconds) containing start
%and end times of every "white" interval (where the stimulus is displayed)
properintervals = properintervals ./ adfreq;


%looking at the length of each proper interval, almost all are exactly 5
%seconds except one anomaly between 1500 and 1600 seconds
a = properintervals(:, 2);
b = properintervals(:, 1);
times = a - b;


%plot the anomaly region's photocell response
plot(ad(1500*adfreq:1600*adfreq))

%plot the start and end times of the photocell "white" periods alongside
%the datarecord start times for each trial to get rid of the problematic
%trials

%{
startsheight = zeros(size(properintervals(:,1))) + 1;
endsheight = zeros(size(properintervals(:,2))) + 1;


scatter(properintervals(:,1), startsheight, 'filled', 'g');
hold on;
scatter(properintervals(:,2), endsheight, 'filled', 'b');

datarecord_starttimes = [];

for i = 1:numel(dataRecord)
    datarecord_starttimes(end+1) = dataRecord(i).elapsedSecs;
end


datarecordheight = zeros(size(datarecord_starttimes)) + 1;
hold on;
scatter(datarecord_starttimes, datarecordheight, 'filled', 'r');

%}

%anomalies: at around 450-460 seconds nothing happens for one cycle,
%resumes as normal exactly the next cycle
% same happens between 930-940 seconds and 1430-1440 seconds
% around 1540 seconds the weird thing happens; no disruption in the
% datarecord times, but essentially get 3 photocell cycles where there
% should be two, goes back to normal after
% around 1930-1940 seconds the photocell skips a cycle, right after the
% datarecord skips a cycle, back on track just before 1960 seconds
% photocell cycle ends early, at 2470-2480 seconds have two datarecord
% cycles with no photocell cycle

% problematic datarecord cycles:
% 1532 seconds ((12, 4)) has a shortened photocell cycle, 1540 seconds ((13,4)) starts
% in the middle of a photocell cycle
% 1934 seconds, 1943 seconds ( (59, 4) and (60, 4) ) have a weird photocell
% cycle (not aligned)
%2470 seconds, 2480 seconds ( (59,5) and (60, 5) ) have no photocell cycle

disallowed_floors = [1532, 1540, 1934, 1943, 2468, 2477];

starttimes = properintervals(:,1);
endtimes = properintervals(:,2);

%if trial is not disallowed, then add trial data (movie name, datarecord
%start time, photocell start and end times) to a table

organizedstarttimes = [];
organizedendtimes = [];
datarecordtimes = [];
movienames = [""];

for c = 1:5
    for r = 1:60
        if ~ismember(floor(dataRecord(r,c).elapsedSecs), disallowed_floors)
            photocellstart = min(starttimes(starttimes>=dataRecord(r,c).elapsedSecs));
            photocellend = min(endtimes(endtimes>=dataRecord(r,c).elapsedSecs));
            organizedstarttimes = [organizedstarttimes; [photocellstart]];
            organizedendtimes = [organizedendtimes; [photocellend]];
            datarecordtimes = [datarecordtimes; [dataRecord(r,c).elapsedSecs]];
            movienames(end+1) = dataRecord(r,c).mvFileName;
        end
    end
end

movienames(1) = [];
movienames = movienames';

%loop through the spike times and sort them by which trial they fall under,
%add them all into an array in the corresponding trial
emptycol = cell(numel(movienames), 1);

for i = ts'
    for j = 1:numel(organizedstarttimes)
        if (organizedstarttimes(j) <= i) & (organizedendtimes(j) >= i)
            emptycol{j,1}(end+1) = i;
        end
    end
end

finaltable = table(organizedstarttimes, organizedendtimes, datarecordtimes, movienames, emptycol);

writetable(finaltable, "exported_neuron.csv")

meanluminance = mean(mvMovie, "all");