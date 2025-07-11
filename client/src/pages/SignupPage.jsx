// SignupPage.jsx
import { useState } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { register } from '../store/slices/authSlice'
import { useNavigate, Link } from 'react-router-dom'
import ThemeToggle from '../components/ThemeToggle'
import healthcareImg from '../assets/healthcare.jpg';

export default function SignupPage() {
  const dispatch = useDispatch()
  const navigate = useNavigate()
  const { loading, error } = useSelector((state) => state.auth)

  const [form, setForm] = useState({ name: '', email: '', password: '' })
  const [showPassword, setShowPassword] = useState(false)

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value })
  }

  const handleRegister = (e) => {
    e.preventDefault()
    dispatch(register(form)).then((res) => {
      if (res.meta.requestStatus === 'fulfilled') navigate('/chat')
    })
  }

  const handleCallAmbulance = () => {
    window.open('tel:102')
  }

  const handleNearbyHospitals = () => {
    window.open('https://www.google.com/maps/search/nearby+hospitals')
  }

  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-br from-white to-indigo-100 dark:from-gray-900 dark:to-gray-900">
      {/* Header Navbar */}
      <header className="w-full flex justify-between items-center px-6 py-4 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 shadow-md">
        <h1 className="text-xl font-bold text-indigo-600 dark:text-indigo-300">Health Assistant</h1>
        <div className="flex items-center gap-4">
          <button
            onClick={handleCallAmbulance}
            className="text-sm px-4 py-2 bg-red-100 text-red-600 font-semibold rounded hover:bg-red-200 transition"
          >
            🚑 Call Ambulance
          </button>
          <button
            onClick={handleNearbyHospitals}
            className="text-sm px-4 py-2 bg-green-100 text-green-600 font-semibold rounded hover:bg-green-200 transition"
          >
            🏥 Nearby Hospitals
          </button>
          <div className="ml-auto">
            <ThemeToggle />
          </div>
        </div>
      </header>

      <div className="flex flex-1 w-full">
        {/* Left Section - Signup Form */}
        <div className="w-full md:w-2/3 flex flex-col justify-center bg-transparent px-8 sm:px-16 md:px-24">
          <div className="bg-white dark:bg-gray-800 p-8 rounded-xl shadow-lg w-full">
            <h2 className="text-3xl font-bold text-indigo-600 dark:text-indigo-300 mb-6 text-center">
              Create an Account
            </h2>

            <form onSubmit={handleRegister} className="space-y-5">
              <div>
                <label className="block mb-1 text-sm font-medium text-gray-700 dark:text-gray-300">Name</label>
                <input
                  type="text"
                  name="name"
                  value={form.name}
                  onChange={handleChange}
                  className="w-full px-4 py-3 rounded-lg border border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white focus:ring-2 focus:ring-indigo-500"
                  placeholder="Your name"
                  required
                />
              </div>

              <div>
                <label className="block mb-1 text-sm font-medium text-gray-700 dark:text-gray-300">Email</label>
                <input
                  type="email"
                  name="email"
                  value={form.email}
                  onChange={handleChange}
                  className="w-full px-4 py-3 rounded-lg border border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white focus:ring-2 focus:ring-indigo-500"
                  placeholder="Your email"
                  required
                />
              </div>

              <div>
                <label className="block mb-1 text-sm font-medium text-gray-700 dark:text-gray-300">Password</label>
                <div className="relative">
                  <input
                    type={showPassword ? 'text' : 'password'}
                    name="password"
                    value={form.password}
                    onChange={handleChange}
                    className="w-full px-4 py-3 rounded-lg border border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white focus:ring-2 focus:ring-indigo-500"
                    placeholder="••••••••"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute top-2.5 right-3 text-sm text-indigo-500 dark:text-indigo-300"
                  >
                    {showPassword ? 'Hide' : 'Show'}
                  </button>
                </div>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full bg-indigo-600 hover:bg-indigo-700 text-white py-3 rounded-lg transition font-semibold"
              >
                {loading ? 'Registering...' : 'Sign Up'}
              </button>

              {error && <p className="text-red-500 text-sm text-center">{error}</p>}

              <p className="text-center text-sm text-gray-600 dark:text-gray-400">
                Already have an account?{' '}
                <Link to="/login" className="text-indigo-600 hover:underline">
                  Login
                </Link>
              </p>
            </form>
          </div>
        </div>

        {/* Right Section - Illustration and Tips */}
         <div className="hidden md:flex flex-col justify-center items-center w-1/3 bg-transparent p-10 space-y-4 transition-colors duration-300">
                 <div className="bg-white/30 dark:bg-white/10 backdrop-blur-lg p-4 rounded-2xl shadow-lg">
         <img
           src={healthcareImg}
           alt="Health AI"
           className="w-44 h-auto rounded-xl shadow-xl transform transition-transform duration-300 hover:scale-105 hover:rotate-1"
         />
       </div>
          <h3 className="text-xl font-bold text-indigo-700 dark:text-white">Your Health Matters</h3>
          <ul className="list-disc list-inside text-[16px] text-gray-700 dark:text-gray-300 space-y-2 font-semibold">
            <li>Get symptom-based diagnosis instantly</li>
            <li>Find nearby hospitals on the map</li>
            <li>Call ambulance in 1 click</li>
            <li>AI answers for your health queries</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
