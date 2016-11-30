function h = confmatrix(probaR , classe_name,Title)
%
% Display Confusion Matrix
%
% Usage
% ------
% 
% h = confmatrix(probaR , [classe_name]);
% 
% 
% Inputs
% -------
%  
% probaR            Confusion matrix (d x d)
% classe_name       Label names (1 x d) cell vector
%
%
% Output
% ------
%
% h                 Handle on figure associated with confusion matrix
% 
%
%
%
% Author : Sébastien PARIS : sebastien.paris@lsis.org
% -------  Date : 10/13/2009


[d , dd] = size(probaR);

if (d ~= dd)
    error('probaR must be a square matrix')
    
end

if(nargin < 2)
    
    classe_name = [];
    
end

diagR         = diag(probaR);
temp          = probaR - diag(diagR);
probasort     = sort(unique(temp(:)));
lp            = length(probasort);

thresh_proba1 = probasort(ceil(0.80*lp));
thresh_proba2 = min(diagR);


[ip1 , jp1]   = find((probaR > thresh_proba1) & (probaR <thresh_proba2) );
ind1          = ip1 + (jp1-1)*d;

[ip2 , jp2]   = find(probaR >= thresh_proba2);
ind2          = ip2 + (jp2-1)*d;

figure,
h             =  imagesc(probaR);
axis ij
colormap(flipud(gray))


title(Title,'FontSize', 10,'FontWeight','bold')

set(gca,'yTick',1:d , 'xTick' , 1:d);

if(~isempty(classe_name) && iscell(classe_name))
    set(gca,'XTickLabel',classe_name(1:1:d));
    aa=get(gca,'XTickLabel');
    bb=get(gca,'XTick');
    cc=get(gca,'YTick');
    th=text(bb,repmat(cc(end)+.6*(cc(2)-cc(1)),length(bb),1),aa,'HorizontalAlignment','left','rotation',310);
    set(th , 'fontsize' , 8) % axis x labels
    set(th , 'Color' , 'k')
    set(gca,'XTickLabel',{});
    set(gca,'yTickLabel',classe_name(1:1:d));
    set(gca , 'fontsize' , 8) % axis y labels
end

showValues=1;
if showValues==1
    h = text(ip1-0.325 , jp1 , num2str(probaR(ind1) , '%2.0f'));
    %h = text(jp1-0.325 , ip1 , num2str(probaR(ind1) , '%2.0f'));
    
    set(h , 'fontsize' , 8 , 'color' , [ 0 0 0])
    
    %h = text(ip2-0.325 , jp2 , num2str(probaR(ind2) , '%3.2f'));
    h = text(jp2-0.325 , ip2 , num2str(probaR(ind2) , '%2.0f'));
    
    set(h , 'fontsize' , 8 , 'color' , [ 1 1 1])
    
end
% xlabel('Ranking', 'fontsize',11,'fontweight','b')
% ylabel('Ranking', 'fontsize',11,'fontweight','b')



