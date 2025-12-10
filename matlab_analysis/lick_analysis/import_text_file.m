%% Import data from text file

function t = import_text_file(filepath)

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 2);

% Specify range and delimiter
% opts.DataLines = [5, Inf];
opts.DataLines = [3, Inf];
opts.Delimiter = " ";

% Specify column names and types
opts.VariableNames = ["Time", "Event"];
opts.VariableTypes = ["double", "categorical"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts.ConsecutiveDelimitersRule = "join";
opts.LeadingDelimitersRule = "ignore";

% Specify variable properties
opts = setvaropts(opts, "Event", "EmptyFieldRule", "auto");

% Import the data
t = readtable(filepath, opts);


%% Clear temporary variables
clear opts