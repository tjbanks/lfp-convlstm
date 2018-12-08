function seg = getseg(LFP,xcld,minLen)
LFP = reshape(LFP,[],1);
T = length(LFP);

alt0 = diff(LFP==0);
art = reshape(find(alt0),1,[]);
if alt0(art(1))==-1
    art = [0,art];
end
if alt0(art(end))==1
    art = [art,T];
end
art = reshape(art,2,[]);
dur = art(2,:)-art(1,:);
art = art(:,dur>=xcld);

seg = reshape(art,1,[]);
if seg(1)==0
    seg = seg(2:end);
else
    seg = [0,seg];
end
if seg(end)==T
    seg(end) = [];
else
    seg = [seg,T];
end
seg = reshape(seg,2,[]);
segdur = seg(2,:)-seg(1,:);
seg = seg(:,segdur>=minLen);
%seg(1,:) = seg(1,:)+1;
end