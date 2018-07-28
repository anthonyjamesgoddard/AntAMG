/* The following code implements an Algebraic Multigrid algorithm.
    The algorithm was pseudo-coded in Ruge and Struben. We attempt a 
    relatively modern C++ implementation here.
    

    We favour std::vector over std::set as std::vector is more optimised
    and is pretty much always recommended over std::set : See ""Professional C++"
    
    I use tabs instead of spaces. This might look strange in git.

*/



#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <numeric>
using Eigen::MatrixXd;
using Eigen::VectorXd;
typedef std::vector<int> intContainer;


//                                                  //
//      Helper functions used in the algorithm.     //
//                                                  //

/* External sort function */
bool orderByCardinality(const std::pair<int,double>&a,const std::pair<int,double>&b)
{
    return a.second<b.second;
}

/* Determines whether one vector is a subset of another.
    We do not have copies of items in a vector so we do
    not need to worry about these cases.         */
bool isSubset(std::vector<int> A, std::vector<int> B)
{
    std::sort(A.begin(), A.end());
    std::sort(B.begin(), B.end());

    /* Returns true if every element in the range [ B.begin(), B.end() )
        is contained in the range [A.begin(), A.end() ).

        i.e A contains B OR B is a subset of A. */
    return std::includes(A.begin(), A.end(), B.begin(), B.end());
}

/* Returns the difference between two vectors.
    The elements in v1 that are not in v2. */
std::vector<int> difference(std::vector<int> v1, std::vector<int> v2)
{
    std::sort(v1.begin(),v1.end());
    std::sort(v2.begin(),v2.end());
    std::vector<int> diff;
    std::set_difference(v1.begin(), v1.end(), v2.begin(), v2.end(),std::inserter(diff, diff.begin()));
    return diff;
}

/* Returns the intersection between two vectors. */
std::vector<int> intersection(std::vector<int> v1, std::vector<int> v2)
{
    std::sort(v1.begin(),v1.end());
    std::sort(v2.begin(),v2.end());
    std::vector<int> intsec(v1.size()+v2.size());
    auto it = std::set_intersection(v1.begin(),v1.end(),v2.begin(),v2.end(),intsec.begin());
    intsec.resize(it-intsec.begin());
    return intsec;
}

/* Algorithm A1,A2,A3 broken up into different methods. */


/* Obtains the connectivity matrix of a matrix. */
void getConnectivityMatrixAndNeighbourhood(MatrixXd& input,
                            const int& N, 
                            double& theta, 
                            std::vector<intContainer>& S,
                            std::vector<intContainer>& St,
                            std::vector<intContainer>& Nbrs);

void getInitialCoarseAndFinePoints(std::vector<intContainer>& S,
                                std::vector<intContainer>& St,
                                const int& N, 
                                std::vector<int>& C,
                                std::vector<int>& F);
                                
void getFinalCPointChoiceAndInterpolationWeights(MatrixXd&input,
                                                std::vector<int>&C,
                                                std::vector<int>&F,
                                                const int &N,
                                                MatrixXd& weights,
                                                std::vector<intContainer>&S,
                                                std::vector<intContainer>&Nbrs);
int main()
{
    const int N = 22;
    double theta = 0.25;
    MatrixXd testingMatrix;
    MatrixXd weights;

    /* Element i of S contains the indices of the points
        which point i is strongly connected to. Element i
        of St contains the points that are strongly connected 
        to i. There is a difference.  */
       
    std::vector<intContainer> S(N),St(N),Nbrs(N);
    std::vector<int> C,F;
    /* Random matrix that we will test the code with.*/
    testingMatrix = Eigen::MatrixXd::Random(N,N);
    weights =       Eigen::MatrixXd::Zero(N,N);

    /* Obtain a connectivity matrix.
        This is acheived by a getter function
        that fills our containers. */
    getConnectivityMatrixAndNeighbourhood(testingMatrix,N,theta,S,St,Nbrs);
    /* Algorithm A2: Ruge, Struben */
    getInitialCoarseAndFinePoints(S,St,N,C,F);
    /* Algorithm A3: Ruge, Struben */
    getFinalCPointChoiceAndInterpolationWeights(testingMatrix,C,F,N,weights,S,Nbrs);



}

/* Obtains the connectivity matrix of a matrix. */
void getConnectivityMatrixAndNeighbourhood(MatrixXd& input,
                            const int& N, 
                            double& theta, 
                            std::vector<intContainer>& S,
                            std::vector<intContainer>& St,
                            std::vector<intContainer>& Nbrs)
{
    S.clear();
    St.clear();
    VectorXd maxRow = (-1*input).rowwise().maxCoeff();

    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            if(i==j) continue;
            if(-1*input(i,j) > theta*maxRow(i))
            {
                S[i].push_back(j);
                St[j].push_back(i);
            }
        }
    }
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            if(i==j) continue;
            if(input(i,j))
            {
                Nbrs[i].push_back(j);
            }
        }
    }
}

void getInitialCoarseAndFinePoints(std::vector<intContainer>& S,
                                std::vector<intContainer>& St,
                                const int& N, 
                                std::vector<int>& C,
                                std::vector<int>& F)
{
    //                  //
    //      STEP 1      //
    //                  //
    std::vector<std::pair<int,double>> lambda;
    
    /* C (coarse) and F (fine) will be filled by our algorithm. */
    for(int i=0;i<N;i++) lambda.push_back(std::make_pair(i,std::accumulate(St[i].begin(), St[i].end(), 0)));
    /* Sorts with respect to the cardinalities: Smallest to largest. */
    std::sort(lambda.begin(),lambda.end(),orderByCardinality);

    //                  //
    //      STEP 2      //
    //                  //

    while(lambda.size() >0)
    {
        /* Pick an i in U with max... */
        std::pair<int,double> lambda_current = lambda.back();
        lambda.pop_back();

        /* To make the code cleaner. */
        int i = lambda_current.first;
        double iCard = lambda_current.second;
        C.push_back(i);

        //                  //
        //      STEP 3      //
        //                  //
        for(int k=0;k < lambda.size();k++)
        {
            //                  //
            //      STEP 4      //
            //                  //
            int j= lambda[k].first;
            if(std::find(St[i].begin(),St[i].end(),j) == St[i].end()) continue;
            // F = F U {j}
            F.push_back(j);
            // Essentially: U=U-{j}
            lambda.erase(lambda.begin() + k);
            for(int m=0;m<lambda.size();m++)
            {
                //                  //
                //      STEP 5      //
                //                  //
                int l= lambda[m].first;
                if(std::find(S[j].begin(),S[j].end(),l) == S[j].end()) continue;
                lambda[l].second++;
            }
        }
        for(int k=0;k < lambda.size();k++)
        {
            //                  //
            //      STEP 6      //
            //                  //
            int j= lambda[k].first;
            if(std::find(S[i].begin(),S[i].end(),j) == S[i].end()) continue;
            lambda[j].second--;
        }
    }
}


void getFinalCPointChoiceAndInterpolationWeights(MatrixXd&input,
                                                std::vector<int>&C,
                                                std::vector<int>&F,
                                                const int &N,
                                                MatrixXd& weights,
                                                std::vector<intContainer>&S,
                                                std::vector<intContainer>&Nbrs)
{
    //                  //
    //      STEP 1      //
    //                  //

    std::vector<int> T;
    std::vector<double> d(N);

    //                  //
    //      STEP 2      //
    //                  //  

    while(!isSubset(T,F))
    {
        std::vector<int> diffBetweenFT = difference(F,T);
        for(int l=0;l< diffBetweenFT.size();l++)
        {

            //                  //
            //      STEP 2      //
            //                  //
            stage2:
            int i = diffBetweenFT[l];
            T.push_back(i);
            
            //                  //
            //      STEP 3      //
            //                  //  
            auto Ci = intersection(S[i],C);         /* Interpolatory points. */
            auto Dis = difference(S[i] ,Ci);        /* Non-interpolatory strong connection. */
            auto Diw = difference(Nbrs[i],S[i]);    /* Non-inerpolatory weak connection. */

            std::vector<int> Ctilde;

            //                  //
            //      STEP 4      //
            //                  //  
            stage4:
            d[i] = input(i,i);
            for(int mm=0;mm<Diw.size();mm++)
            {
                int rr = Diw[mm];
                d[rr]+=input(i,rr);
            }

            for(int nn=0;nn<Ci.size();nn++)
            {
                int kk = Ci[nn];
                d[kk] = input(i,kk);
            }

            //                  //
            //      STEP 5      //
            //                  //  

            for(int m=0;m<Dis.size();m++)
            {
                int j = Dis[m];
                int skipAndGoToStep8Flag = 0;

                //                  //
                //      STEP 6      //
                //                  //
                if(intersection(S[j],Ci).size()) skipAndGoToStep8Flag = 1;

                //                  //
                //      STEP 7      //
                //                  //
                if(!skipAndGoToStep8Flag)
                {
                    if(Ctilde.size())
                    {
                        C.push_back(i);
                        auto it = std::find(F.begin(),F.end(),i);
                        if(it !=F.end())
                            F.erase(it);
                        if(l<diffBetweenFT.size())
                        {
                            l++;
                            goto stage2;            /* It seemed that this was a good time to
                                                        use goto. Feel free to refactor */
                        }
                    }
                    else 
                    {
                        Ctilde.clear();
                        Ctilde.push_back(j);
                        if(l<diffBetweenFT.size())
                        {
                            l++;
                            i = diffBetweenFT[l];
                            goto stage4;
                        }
                    }
                }
                //                  //
                //      STEP 8      //
                //                  //
                
                for(int h=0;h<Ci.size();h++)
                {
                    int k = Ci[h];
                    double sum = 0;
                    for(int z=0;z<Ci.size();z++)
                    {
                        int ll = Ci[z];
                        sum+=input(j,ll);
                    }
                    d[k] += input(i,j)*input(j,k)/sum;
                }
            }
            //                  //
            //      STEP 9      //
            //                  //
            for(int h=0;h<Ctilde.size();h++)
            {
                C.push_back(Ctilde[h]);
            }
            F = difference(F,Ctilde);
            for(int h=0;h<Ci.size();h++)
            {
                int k=Ci[h];
                weights(i,k) = -1.0*d[k]/d[i];
            }
            if(l<diffBetweenFT.size())
            {
                l++;
                goto stage2;            /* It seemed that this was a good time to
                                            use goto. Feel free to refactor */
            }
        }
    }
}
