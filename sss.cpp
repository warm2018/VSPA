#include <iostream>
using namespace std;
namespace A
{
    int a;
    int b; 
}

namespace B
{
	int a;
	int b;
}

using namespace B;

/*
void f()
{
	using namespace std;
	cout<<"a"<<a<<endl;
	using A::a;
	a=1;
	cout<<"A::a="<<a<<endl;
	b=3;
}

int main()
{
	using std::cout;
	a=2;
	f();
	cout<<"B::a="<<a<<std::endl;
	cout<<"B::b="<<b<<std::endl;
}
*/

int main()
{
	using namespace A;
	a=1;
	b=2;
}
