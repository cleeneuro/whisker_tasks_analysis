function [aux0, aux1, aux2, aux3, metadata] = read_aux_triggers(tifpath)

%   Reads the timestamps from aux triggers 0-3 for each frame from the
%   ScanImage tiff metadata and makes a plot
%   INPUT: tifpath - path to scanimage tiff stack

% Aux0 = air compressor
% Aux1 = CS plus
% Aux2 = CS minus 
% Aux3 = lick
% the above assumes the arduino is configured in a certain way 

%load metadata
metadata=imfinfo(tifpath);

for f = 1:length(metadata)
    % ImageDescription is a string with all the metadata for each frame
    d = metadata(f).ImageDescription;
    
    % Initialize aux variables as empty arrays
    aux0{f} = [];
    aux1{f} = [];
    aux2{f} = [];
    aux3{f} = [];
    
    % Use regular expressions to extract the values between brackets
    aux0Str = regexp(d, 'auxTrigger0 = \[(.*?)\]', 'tokens', 'once');
    aux1Str = regexp(d, 'auxTrigger1 = \[(.*?)\]', 'tokens', 'once');
    aux2Str = regexp(d, 'auxTrigger2 = \[(.*?)\]', 'tokens', 'once');
    aux3Str = regexp(d, 'auxTrigger3 = \[(.*?)\]', 'tokens', 'once');
    
    % Convert the extracted strings to numbers, if not empty, and assign to cell array
    if ~isempty(aux0Str)
        aux0{f} = str2num(aux0Str{1});
    end
    if ~isempty(aux1Str)
        aux1{f} = str2num(aux1Str{1});
    end
    if ~isempty(aux2Str)
        aux2{f} = str2num(aux2Str{1});
    end
    if ~isempty(aux3Str)
        aux3{f} = str2num(aux3Str{1});
    end
end

nonEmptyCells0 = ~cellfun(@isempty, aux0);
nonEmptyValues_aux0 = aux0(nonEmptyCells0);
fh = figure(1);
subplot(2,2,1)
plot(nonEmptyCells0)
hold on
title('Aux0 - air compressor')
ylim([-1 2])

nonEmptyCells1 = ~cellfun(@isempty, aux1);
nonEmptyValues_aux1 = aux1(nonEmptyCells1);
subplot(2,2,2)
plot(nonEmptyCells1)
hold on
title('Aux1 - CSplus')
ylim([-1 2])


nonEmptyCells2 = ~cellfun(@isempty, aux2);
nonEmptyValues_aux2 = aux2(nonEmptyCells2);
subplot(2,2,3)
plot(nonEmptyCells2)
hold on
title('Aux2 - CSminus')
ylim([-1 2])


nonEmptyCells3 = ~cellfun(@isempty, aux3);
nonEmptyValues_aux3 = aux3(nonEmptyCells3);
subplot(2,2,4)
plot(nonEmptyCells3)
hold on
title('Aux3 - lick')
ylim([-1 2])

han=axes(fh,'visible','off'); 
han.XLabel.Visible='on';
han.YLabel.Visible='on';
ylabel(han, 'TTL input')
xlabel(han,'frames')

